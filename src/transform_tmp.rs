//! Transformation stuff (legacy)

use crate::alpha::handle_extra_channels;
use crate::convert::link_profiles;
use crate::mlu::MLU;
use crate::named::NamedColor;
use crate::optimization::optimize_pipeline;
use crate::pack::{
    formatter, formatter_is_float, Accum, Formatter, FormatterDir, PACK_FLAGS_16BITS,
    PACK_FLAGS_FLOAT,
};
use crate::pcs::lcms_color_space;
use crate::pipe::Pipeline;
use crate::profile::Profile;
use std::fmt;
use crate::white_point::D50;
use crate::{ColorSpace, ICCTag, Intent, PixelFormat, PixelType, ProfileClass, Technology, CIEXYZ};

const DEFAULT_OBSERVER_ADAPTATION_STATE: f64 = 1.;

// TODO: figure out what adaptation states are

// TODO: figure out what alarm codes are for

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stride {
    bytes_per_line_in: u32,
    bytes_per_line_out: u32,
    pub bytes_per_plane_in: u32,
    pub bytes_per_plane_out: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct Cache {
    cache_in: [u16; 16],
    cache_out: [u16; 16],
}

#[derive(Debug, Clone)]
struct SeqItem {
    device_mfg: u32,
    device_moedl: u32,
    attributes: u64,
    technology: Technology,
    profile_id: u128,
    manufacturer: MLU,
    model: MLU,
    description: MLU,
}

type TransformFn = unsafe fn(&Transform, *const (), *mut (), usize, usize, Stride);

#[derive(Clone)]
pub struct Transform {
    pub(crate) input_format: PixelFormat,
    pub(crate) output_format: PixelFormat,

    xform: TransformFn,

    from_input: Formatter,
    to_output: Formatter,

    /// 1-pixel cache
    cache: Cache,

    /// A Pipeline holding the full (optimized) transform
    lut: Pipeline,

    /// A Pipeline holding the gamut check. It goes from the input space to bilevel
    gamut_check: Option<Pipeline>,

    input_colorant: Option<Vec<NamedColor>>,
    output_colorant: Option<Vec<NamedColor>>,

    entry_color_space: Option<ColorSpace>,
    exit_color_space: Option<ColorSpace>,

    entry_white_point: Option<CIEXYZ>,
    exit_white_point: Option<CIEXYZ>,

    /// Profiles used to create the transform
    sequence: Vec<SeqItem>,

    pub(crate) dw_original_flags: TransformFlags,
    adaptation_state: f64,

    /// The intent of this transform. That is usually the last intent in the profile chain, but may differ
    intent: Intent,
}

impl Transform {
    fn alloc_empty(
        lut: Pipeline,
        intent: Intent,
        input_format: PixelFormat,
        output_format: PixelFormat,
        mut dw_flags: TransformFlags,
    ) -> Result<Transform, String> {
        let (xform, from_input, to_output) = if formatter_is_float(input_format as u32)
            && formatter_is_float(output_format as u32)
        {
            // this is a true floating-point transform

            let inf =
                if let Some(inf) = formatter(input_format, FormatterDir::Input, PACK_FLAGS_FLOAT) {
                    inf
                } else {
                    return Err("No such input format".into());
                };
            let out = if let Some(out) =
                formatter(output_format, FormatterDir::Output, PACK_FLAGS_FLOAT)
            {
                out
            } else {
                return Err("No such output format".into());
            };

            dw_flags |= TransformFlags::__CAN_CHANGE_FORMATTER;

            let xform = if dw_flags.contains(TransformFlags::NULLTRANSFORM) {
                null_float_xform
            } else {
                float_xform
            };

            (xform, inf, out)
        } else {
            let (inf, out) = if input_format == 0 && output_format == 0 {
                dw_flags |= TransformFlags::__CAN_CHANGE_FORMATTER;

                (Formatter::None, Formatter::None)
            } else {
                let inf = if let Some(inf) =
                    formatter(input_format, FormatterDir::Input, PACK_FLAGS_16BITS)
                {
                    inf
                } else {
                    return Err("No such input format".into());
                };
                let out = if let Some(out) =
                    formatter(output_format, FormatterDir::Output, PACK_FLAGS_16BITS)
                {
                    out
                } else {
                    return Err("No such output format".into());
                };

                let bppix_in = t_bytes!(input_format);
                if bppix_in == 0 || bppix_in > 2 {
                    dw_flags |= TransformFlags::__CAN_CHANGE_FORMATTER;
                }

                (inf, out)
            };

            let xform = if dw_flags.contains(TransformFlags::NULLTRANSFORM) {
                null_xform
            } else {
                match (
                    dw_flags.contains(TransformFlags::NOCACHE),
                    dw_flags.contains(TransformFlags::GAMUTCHECK),
                ) {
                    (true, true) => precalculated_xform_gamut_check,
                    (true, false) => precalculated_xform,
                    (false, true) => cached_xform_gamut_check,
                    (false, false) => cached_xform,
                }
            };

            (xform, inf, out)
        };

        let transform = Transform {
            from_input,
            to_output,
            input_format,
            output_format,
            lut,
            sequence: Vec::new(),
            dw_original_flags: dw_flags,
            adaptation_state: 0.,
            intent,
            xform,
            cache: Cache::default(),
            entry_color_space: None,
            exit_color_space: None,
            entry_white_point: None,
            exit_white_point: None,
            input_colorant: None,
            output_colorant: None,
            gamut_check: None,
        };

        optimize_pipeline(
            &transform.lut,
            intent,
            input_format,
            output_format,
            dw_flags,
        );

        Ok(transform)
    }

    pub fn new_extended(
        profiles: &[Profile],
        bpc: &[bool],
        intents: &[Intent],
        adaptation_states: &[f64],
        gamut_profile: Option<&Profile>,
        _gamut_pcs_position: u32,
        input_format: u32,
        output_format: u32,
        mut dw_flags: TransformFlags,
    ) -> Result<Transform, String> {
        // If it is a fake transform
        if dw_flags.contains(TransformFlags::NULLTRANSFORM) {
            unimplemented!("alloc_empty_transform")
        }

        // If gamut check is requested, make sure we have a gamut profile
        if dw_flags.contains(TransformFlags::GAMUTCHECK) {
            if gamut_profile.is_none() {
                dw_flags |= !TransformFlags::GAMUTCHECK;
            }
        }

        // On floating point transforms, inhibit cache
        if t_float!(input_format) != 0 || t_float!(output_format) != 0 {
            dw_flags |= TransformFlags::NOCACHE;
        }

        // Mark entry/exit spaces
        let (entry_cs, exit_cs) = match transform_color_spaces(profiles) {
            Some(cs) => cs,
            None => return Err("No color spaces".into()), // TODO: error
        };

        if !is_proper_color_space(entry_cs, input_format) {
            return Err("Entry CS is not proper CS".into()); // TODO: error
        }

        if !is_proper_color_space(exit_cs, output_format) {
            return Err("Exit CS is not proper CS".into()); // TODO: error
        }

        let lut = link_profiles(profiles, intents, bpc, adaptation_states, dw_flags)?;

        // check channel count
        if entry_cs.channels() != lut.input_channels()
            || exit_cs.channels() != lut.output_channels()
        {
            return Err("Channel count doesn’t match. Profile is corrupted".into());
        }

        let last_intent = match intents.last() {
            Some(intent) => *intent,
            None => return Err("No last intent".into()), // TODO: error
        };

        // all seems ok
        let mut transform =
            Transform::alloc_empty(lut, last_intent, input_format, output_format, dw_flags)?;

        // keep values
        transform.entry_color_space = Some(entry_cs);
        transform.exit_color_space = Some(exit_cs);
        transform.intent = last_intent;

        // take white points
        transform.entry_white_point = Some(white_point(
            profiles
                .first()
                .unwrap()
                .read_tag_clone(ICCTag::MediaWhitePoint),
        ));
        transform.exit_white_point = Some(white_point(
            profiles
                .last()
                .unwrap()
                .read_tag_clone(ICCTag::MediaWhitePoint),
        ));

        // create a gamut check LUT if requested
        if let Some(_gamut_profile) = gamut_profile {
            if dw_flags.contains(TransformFlags::GAMUTCHECK) {
                transform.gamut_check = Some(Pipeline::new_gamut_check(
                    profiles,
                    bpc,
                    intents,
                    adaptation_states,
                ));
            }
        }

        // try to read input and output colorant table
        if let Some(col_table) =
            profiles[0].read_tag_clone::<Vec<NamedColor>>(ICCTag::ColorantTable)
        {
            transform.input_colorant = Some(col_table);
        }

        let last_profile = profiles.last().unwrap();
        if last_profile.device_class == ProfileClass::Link {
            // this tag may only exist on devicelink profiles
            if let Some(col_table) =
                last_profile.read_tag_clone::<Vec<NamedColor>>(ICCTag::ColorantTableOut)
            {
                transform.output_colorant = Some(col_table);
            }
        } else if let Some(col_table) =
            last_profile.read_tag_clone::<Vec<NamedColor>>(ICCTag::ColorantTable)
        {
            transform.output_colorant = Some(col_table);
        }

        // store the sequence of profiles
        if dw_flags.contains(TransformFlags::KEEP_SEQUENCE) {
            transform.sequence = compile_profile_sequence(profiles);
        }

        // if this is a cached transform, init first value, which is zero (16 bits only)
        if !dw_flags.contains(TransformFlags::NOCACHE) {
            if transform.gamut_check.is_some() {
                transform_one_pixel_with_gamut_check(&mut transform);
            } else {
                // TODO
                // let lut_data = transform.lut.data;
                // transform.lut.eval_16_fn(transform.cache.cache_in, transform.cache.cache_out, lut_data);
            }
        }

        Ok(transform)
    }

    /// Multiprofile transforms: Gamut check is not available here, as it is unclear from which profile the gamut comes.
    pub fn new_multiprofile(
        profiles: &[Profile],
        input_fmt: PixelFormat,
        output_fmt: PixelFormat,
        intent: Intent,
        dw_flags: TransformFlags,
    ) -> Result<Transform, String> {
        let mut bpc = Vec::with_capacity(256);
        let mut intents = Vec::with_capacity(256);
        let mut adaptation_states = Vec::with_capacity(256);

        for _ in profiles {
            bpc.push(dw_flags.contains(TransformFlags::BLACKPOINTCOMPENSATION));
            intents.push(intent);
            adaptation_states.push(-1.); // TODO: check if this is fine
        }

        Self::new_extended(
            profiles,
            &bpc,
            &intents,
            &adaptation_states,
            None,
            0,
            input_fmt,
            output_fmt,
            dw_flags,
        )
    }

    /// Creates a new transform.
    pub fn new_flags(
        input: &Profile,
        in_fmt: PixelFormat,
        output: &Profile,
        out_fmt: PixelFormat,
        intent: Intent,
        dw_flags: TransformFlags,
    ) -> Result<Transform, String> {
        let profiles = vec![input.clone(), output.clone()];

        Self::new_multiprofile(&profiles, in_fmt, out_fmt, intent, dw_flags)
    }

    /// Creates a new transform with no flags.
    pub fn new(
        input: &Profile,
        in_fmt: PixelFormat,
        output: &Profile,
        out_fmt: PixelFormat,
        intent: Intent,
    ) -> Result<Transform, String> {
        Self::new_flags(
            input,
            in_fmt,
            output,
            out_fmt,
            intent,
            TransformFlags::empty(),
        )
    }

    // TODO: nice API

    /// Temporary conversion function
    #[deprecated]
    pub(crate) unsafe fn convert_any(&self, input: *const (), output: *mut (), size: usize) {
        let stride = Stride {
            bytes_per_line_in: 0,
            bytes_per_line_out: 0,
            bytes_per_plane_in: size as u32,
            bytes_per_plane_out: size as u32,
        };
        (self.xform)(self, input, output, size, 1, stride);
    }

    // TODO: (io1) ChangeInterpolationToTrilinear
}

impl fmt::Debug for Transform {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Transform {{ format: {:b} -> {:b}, formatter: {:?} -> {:?}, lut: {:?}, gamut_check: {:?}, colorants: {:?} -> {:?}, CS: {:?} -> {:?}, WP: {:?} -> {:?}, seq: {:?}, intent: {:?} }}", self.input_format, self.output_format, self.from_input, self.to_output, self.lut, self.gamut_check, self.input_colorant, self.output_colorant, self.entry_color_space, self.exit_color_space, self.entry_white_point, self.exit_white_point, self.sequence, self.intent)
    }
}

fn transform_color_spaces(profiles: &[Profile]) -> Option<(ColorSpace, ColorSpace)> {
    let mut cs_in: Option<ColorSpace> = None;
    let mut cs_out: Option<ColorSpace> = None;

    for i in 0..profiles.len() {
        let profile = &profiles[i];

        let l_is_input = if let Some(cs_out) = cs_out {
            cs_out != ColorSpace::XYZ && cs_out != ColorSpace::Lab
        } else {
            true
        };

        let (s_in, s_out) = if profile.device_class == ProfileClass::NamedColor {
            (
                ColorSpace::S1Color,
                if profiles.len() > 1 {
                    profile.pcs
                } else {
                    profile.color_space
                },
            )
        } else if l_is_input || profile.device_class == ProfileClass::Link {
            (profile.color_space, profile.pcs)
        } else {
            (profile.pcs, profile.color_space)
        };

        if i == 0 {
            cs_in = Some(s_in);
        }

        cs_out = Some(s_out);
    }

    match cs_in {
        Some(cs_in) => Some((cs_in, cs_out.unwrap())),
        None => None,
    }
}

fn is_proper_color_space(check: ColorSpace, dw_format: u32) -> bool {
    let space1 = t_colorspace!(dw_format);
    let space2 = lcms_color_space(check);

    if space1 == PixelType::Any as u32 || space1 == space2 as u32 {
        true
    } else if space1 == PixelType::LabV2 as u32 && space2 == PixelType::Lab {
        true
    } else if space1 == PixelType::Lab as u32 && space2 == PixelType::LabV2 {
        true
    } else {
        false
    }
}

// Jun-21-2000: Some profiles (those that come with W2K)
// come with the media white (media black?) x 100. Add a sanity check

fn normalize_xyz(dest: &mut CIEXYZ) {
    while dest.x > 2. && dest.y > 2. && dest.z > 2. {
        dest.x /= 10.;
        dest.y /= 10.;
        dest.z /= 10.;
    }
}

fn white_point(src: Option<CIEXYZ>) -> CIEXYZ {
    match src {
        Some(src) => {
            let mut wp = src;
            normalize_xyz(&mut wp);
            wp
        }
        None => D50,
    }
}

// transform routines

/// Float xform converts floats. Since there are no performance issues, one routine does all job, including gamut check.
/// Note that because extended range, we can use a -1.0 value for out of gamut in this case.
unsafe fn float_xform(
    transform: &Transform,
    input: *const (),
    output: *mut (),
    width: usize,
    height: usize,
    stride: Stride,
) {
    handle_extra_channels(transform, input, output, width, height, stride);

    let mut stride_in = 0;
    let mut stride_out = 0;
    let mut f_in = [0.; 16];
    let mut f_out = [0.; 16];

    let input = Accum::new(input as *mut _);
    let output = Accum::new(output);

    for _ in 0..height {
        let mut accum = input + stride_in;
        let mut output = output + stride_out;

        for _ in 0..width {
            accum = (transform.from_input.unwrap_float())(
                transform,
                &mut f_in,
                accum,
                stride.bytes_per_plane_in,
            );

            // Any gamut chack to do?
            if let Some(ref gamut_check) = transform.gamut_check {
                // Evaluate gamut marker.
                let mut out_of_gamut = [0.; 1];
                gamut_check.eval_float(&f_in, &mut out_of_gamut);

                // Is current color out of gamut?
                if out_of_gamut[0] > 0. {
                    // Certainly, out of gamut
                    for c in 0..16 {
                        f_out[c] = -1.;
                    }
                } else {
                    // No, proceed normally
                    transform.lut.eval_float(&f_in, &mut f_out);
                }
            } else {
                // No gamut check at all
                transform.lut.eval_float(&f_in, &mut f_out);
            }

            output = (transform.to_output.unwrap_float())(
                transform,
                &mut f_out,
                output,
                stride.bytes_per_plane_out,
            );
        }

        stride_in += stride.bytes_per_plane_in;
        stride_out += stride.bytes_per_plane_out;
    }
}

unsafe fn null_float_xform(
    transform: &Transform,
    input: *const (),
    output: *mut (),
    width: usize,
    height: usize,
    stride: Stride,
) {
    handle_extra_channels(transform, input, output, width, height, stride);

    let mut stride_in = 0;
    let mut stride_out = 0;
    let mut f_in = [0.; 16];

    for _ in 0..height {
        let mut accum = Accum::new(input as *mut _) + stride_in;
        let mut output = Accum::new(output) + stride_out;

        for _ in 0..width {
            accum = (transform.from_input.unwrap_float())(
                transform,
                &mut f_in,
                accum,
                stride.bytes_per_plane_in,
            );
            output = (transform.to_output.unwrap_float())(
                transform,
                &mut f_in,
                output,
                stride.bytes_per_plane_out,
            );
        }

        stride_in += stride.bytes_per_plane_in;
        stride_out += stride.bytes_per_plane_out;
    }
}

// TODO
unsafe fn null_xform(_: &Transform, _: *const (), _: *mut (), _: usize, _: usize, _: Stride) {
    unimplemented!()
}
unsafe fn precalculated_xform(
    _: &Transform,
    _: *const (),
    _: *mut (),
    _: usize,
    _: usize,
    _: Stride,
) {
    unimplemented!()
}
unsafe fn cached_xform(_: &Transform, _: *const (), _: *mut (), _: usize, _: usize, _: Stride) {
    unimplemented!()
}
unsafe fn precalculated_xform_gamut_check(
    _: &Transform,
    _: *const (),
    _: *mut (),
    _: usize,
    _: usize,
    _: Stride,
) {
    unimplemented!()
}
unsafe fn cached_xform_gamut_check(
    _: &Transform,
    _: *const (),
    _: *mut (),
    _: usize,
    _: usize,
    _: Stride,
) {
    unimplemented!()
}

fn compile_profile_sequence(_profiles: &[Profile]) -> Vec<SeqItem> {
    // TODO (io1)
    unimplemented!()
}

fn transform_one_pixel_with_gamut_check(_transform: &Transform) {
    unimplemented!()
}
