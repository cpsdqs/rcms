//! Color space transformations.
//!
//! # Examples
//! ```
//! # use lcms_prime::*;
//! # use lcms_prime::pixel_format::*;
//! # fn create_aces_cg_profile_somehow() -> Profile {
//! #     Profile::new_rgb(
//! #         CIExyY { x: 0.32168, y: 0.33767, Y: 1. },
//! #         CIExyYTriple {
//! #             red: CIExyY { x: 0.713, y: 0.293, Y: 1. },
//! #             green: CIExyY { x: 0.165, y: 0.830, Y: 1. },
//! #             blue: CIExyY { x: 0.128, y: 0.044, Y: 1. },
//! #         },
//! #         [
//! #             ToneCurve::new_gamma(1.).unwrap(),
//! #             ToneCurve::new_gamma(1.).unwrap(),
//! #             ToneCurve::new_gamma(1.).unwrap(),
//! #         ],
//! #     ).unwrap()
//! # }
//! // input and output profiles
//! let srgb = Profile::new_srgb();
//! let aces_cg = create_aces_cg_profile_somehow();
//!
//! // create the Transform from sRGB to ACEScg with a perceptual intent
//! let srgb_to_aces: Transform<RGB<f64>, RGB<f32>> =
//!     Transform::new(&srgb, &aces_cg, Intent::Perceptual).unwrap();
//!
//! // RGB<f64> pixel format
//! let srgb_data: [f64; 3] = [0.7, 0.3, 0.1];
//! // RGB<f32> pixel format
//! let mut aces_cg_data: [f32; 3] = [0.; 3];
//!
//! // perform the conversion from sRGB to ACEScg
//! srgb_to_aces.convert(&srgb_data, &mut aces_cg_data);
//!
//! println!("sRGB {:?} -> ACEScg {:?}", srgb_data, aces_cg_data);
//! // => sRGB [0.7, 0.3, 0.1] -> ACEScg [0.3000017, 0.0986936, 0.025978323]
//!
//! const TOLERANCE: f32 = 1e-4;
//! fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < TOLERANCE }
//!
//! // test values from ColorSync Utility
//! assert!(approx_eq(aces_cg_data[0], 0.3000));
//! assert!(approx_eq(aces_cg_data[1], 0.0987));
//! assert!(approx_eq(aces_cg_data[2], 0.0260));
//! ```

use convert::link_profiles;
use lut::Pipeline;
use pixel_format::{DynPixelFormat, PixelFormat, MAX_CHANNELS};
use profile::Profile;
use std::marker::PhantomData;
use std::{fmt, mem, slice};
use transform_tmp::TransformFlags;
use {ColorSpace, Intent, ProfileClass};

type TransformFn<I, O> = fn(transform: &Transform<I, O>, input: &[u8], output: &mut [u8]);
type DynTransformFn = fn(
    transform: &DynTransform,
    in_fmt: &DynPixelFormat,
    out_fmt: &DynPixelFormat,
    input: &[u8],
    output: &mut [u8],
);

// TODO: deduplicate code

/// A dynamic transform; unsafe but supports dynamic pixel formats.
pub struct DynTransform {
    pipeline: Pipeline,
    intent: Intent,
    in_fmt: DynPixelFormat,
    out_fmt: DynPixelFormat,
    transform_fn: DynTransformFn,
}

impl DynTransform {
    fn alloc(
        pipeline: Pipeline,
        intent: Intent,
        in_fmt: DynPixelFormat,
        out_fmt: DynPixelFormat,
    ) -> DynTransform {
        let transform_fn = if in_fmt.is_float && out_fmt.is_float {
            dyn_transform_float
        } else {
            unimplemented!()
        };

        DynTransform {
            pipeline,
            intent,
            in_fmt,
            out_fmt,
            transform_fn,
        }
    }

    /// Creates a new transform with two profiles.
    pub fn new(
        input: &Profile,
        output: &Profile,
        intent: Intent,
        in_fmt: DynPixelFormat,
        out_fmt: DynPixelFormat,
    ) -> Result<DynTransform, String> {
        let profiles = vec![input.clone(), output.clone()];

        Self::new_multi(&profiles, intent, false, in_fmt, out_fmt)
    }

    /// Creates a new transform with arbitrarily many profiles, an intent, and optional black point compensation.
    pub fn new_multi(
        profiles: &[Profile],
        intent: Intent,
        bpc: bool,
        in_fmt: DynPixelFormat,
        out_fmt: DynPixelFormat,
    ) -> Result<DynTransform, String> {
        let mut bpcs = Vec::with_capacity(256);
        let mut intents = Vec::with_capacity(256);
        let mut adaptation_states = Vec::with_capacity(256);

        for _ in profiles {
            // bpc.push(dw_flags.contains(TransformFlags::BLACKPOINTCOMPENSATION));
            bpcs.push(bpc);
            intents.push(intent);
            adaptation_states.push(-1.); // TODO: check if this is fine
        }

        Self::new_ex(
            profiles,
            &bpcs,
            &intents,
            &adaptation_states,
            in_fmt,
            out_fmt,
        )
    }

    /// Creates a new transform.
    ///
    /// profiles, bpc (black point compensation), intents, and adaptation states must all have the
    /// same length.
    pub fn new_ex(
        profiles: &[Profile],
        bpc: &[bool],
        intents: &[Intent],
        adaptation_states: &[f64],
        in_fmt: DynPixelFormat,
        out_fmt: DynPixelFormat,
    ) -> Result<DynTransform, String> {
        // TODO: better errors
        let (entry_cs, exit_cs) = match input_output_spaces(profiles) {
            Some(x) => x,
            None => return Err("Too few profiles in transform".into()),
        };

        if entry_cs != in_fmt.space {
            return Err("Entry color space doesn’t match input format".into());
        }
        if exit_cs != out_fmt.space {
            return Err("Exit color space doesn’t match output format".into());
        }

        let pipeline = link_profiles(
            profiles,
            intents,
            bpc,
            adaptation_states,
            TransformFlags::empty(),
        )?;

        if entry_cs.channels() != pipeline.input_channels() {
            return Err(
                "Entry color space and pipeline input don’t have the same channel count".into(),
            );
        }

        if exit_cs.channels() != pipeline.output_channels() {
            return Err(
                "Exit color space and pipeline output don’t have the same channel count".into(),
            );
        }

        let transform = Self::alloc(pipeline, *intents.last().unwrap(), in_fmt, out_fmt);

        // TODO: gamut check
        // TODO: 16-bit colorant stuff
        // TODO: cache stuff etc.

        Ok(transform)
    }

    /// Converts values in an input buffer to values in the output buffer.
    ///
    /// Lengths are in bytes.
    pub fn convert(&self, input: *const (), input_len: usize, output: *mut (), output_len: usize) {
        let input = unsafe { slice::from_raw_parts(input as *const u8, input_len) };
        let output = unsafe { slice::from_raw_parts_mut(output as *mut u8, output_len) };
        (self.transform_fn)(self, &self.in_fmt, &self.out_fmt, input, output);
    }
}

/// A transform.
pub struct Transform<InFmt: PixelFormat, OutFmt: PixelFormat> {
    pipeline: Pipeline,
    intent: Intent,
    transform_fn: TransformFn<InFmt, OutFmt>,

    _phantom_in: PhantomData<InFmt>,
    _phantom_out: PhantomData<OutFmt>,
}

impl<InFmt: PixelFormat, OutFmt: PixelFormat> Transform<InFmt, OutFmt> {
    fn alloc(pipeline: Pipeline, intent: Intent) -> Transform<InFmt, OutFmt> {
        let transform_fn = if InFmt::IS_FLOAT && OutFmt::IS_FLOAT {
            transform_float
        } else {
            unimplemented!()
        };

        Transform {
            pipeline,
            intent,
            transform_fn,

            _phantom_in: PhantomData,
            _phantom_out: PhantomData,
        }
    }

    /// Creates a new transform with two profiles.
    pub fn new(
        input: &Profile,
        output: &Profile,
        intent: Intent,
    ) -> Result<Transform<InFmt, OutFmt>, String> {
        let profiles = vec![input.clone(), output.clone()];

        Self::new_multi(&profiles, intent, false)
    }

    /// Creates a new transform with arbitrarily many profiles, an intent, and optional black point compensation.
    pub fn new_multi(
        profiles: &[Profile],
        intent: Intent,
        bpc: bool,
    ) -> Result<Transform<InFmt, OutFmt>, String> {
        let mut bpcs = Vec::with_capacity(256);
        let mut intents = Vec::with_capacity(256);
        let mut adaptation_states = Vec::with_capacity(256);

        for _ in profiles {
            // bpc.push(dw_flags.contains(TransformFlags::BLACKPOINTCOMPENSATION));
            bpcs.push(bpc);
            intents.push(intent);
            adaptation_states.push(-1.); // TODO: check if this is fine
        }

        Self::new_ex(profiles, &bpcs, &intents, &adaptation_states)
    }

    /// Creates a new transform.
    ///
    /// profiles, bpc (black point compensation), and intents must all have the same length.
    pub fn new_ex(
        profiles: &[Profile],
        bpc: &[bool],
        intents: &[Intent],
        adaptation_states: &[f64],
    ) -> Result<Transform<InFmt, OutFmt>, String> {
        // TODO: better errors
        let (entry_cs, exit_cs) = match input_output_spaces(profiles) {
            Some(x) => x,
            None => return Err("Too few profiles in transform".into()),
        };

        if entry_cs != InFmt::SPACE {
            return Err("Entry color space doesn’t match input format".into());
        }
        if exit_cs != OutFmt::SPACE {
            return Err("Exit color space doesn’t match output format".into());
        }

        let pipeline = link_profiles(
            profiles,
            intents,
            bpc,
            adaptation_states,
            TransformFlags::empty(),
        )?;

        if entry_cs.channels() != pipeline.input_channels() {
            return Err(
                "Entry color space and pipeline input don’t have the same channel count".into(),
            );
        }

        if exit_cs.channels() != pipeline.output_channels() {
            return Err(
                "Exit color space and pipeline output don’t have the same channel count".into(),
            );
        }

        let transform = Self::alloc(pipeline, *intents.last().unwrap());

        // TODO: gamut check
        // TODO: 16-bit colorant stuff
        // TODO: cache stuff etc.

        Ok(transform)
    }

    /// Converts values in an input buffer to values in the output buffer.
    pub fn convert(&self, input: &[InFmt::Element], output: &mut [OutFmt::Element]) {
        let input = unsafe {
            slice::from_raw_parts(
                input.as_ptr() as *const _,
                input.len() * mem::size_of::<InFmt::Element>(),
            )
        };
        let output = unsafe {
            slice::from_raw_parts_mut(
                output.as_ptr() as *mut _,
                output.len() * mem::size_of::<OutFmt::Element>(),
            )
        };
        (self.transform_fn)(self, input, output);
    }
}

impl<InFmt: PixelFormat, OutFmt: PixelFormat> fmt::Debug for Transform<InFmt, OutFmt> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Transform {{ intent: {:?}, pipeline: {:?} }}",
            self.intent, self.pipeline
        )
    }
}

fn input_output_spaces(profiles: &[Profile]) -> Option<(ColorSpace, ColorSpace)> {
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

/// Handles a transformation using floats.
///
/// The input and output buffers are for all intents and purposes void pointers with lengths.
fn transform_float<I, O>(transform: &Transform<I, O>, input: &[u8], output: &mut [u8])
where
    I: PixelFormat,
    O: PixelFormat,
{
    let px_size_in = I::size();
    let px_size_out = O::size();

    if output.len() / px_size_out < input.len() / px_size_in {
        return; // TODO: error
    }

    // TODO: handle extra channels
    // handle_extra_channels(transform, input as *const _ as *const _, output as *mut _ as *mut _, input.len(), 1, 0);

    let offset_in = if I::EXTRA_FIRST {
        I::EXTRA_CHANNELS * mem::size_of::<I::Element>()
    } else {
        0
    };
    let offset_out = if O::EXTRA_FIRST {
        O::EXTRA_CHANNELS * mem::size_of::<O::Element>()
    } else {
        0
    };

    for i in 0..(input.len() / px_size_in) {
        let pos_in = i * px_size_in + offset_in;
        let pos_out = i * px_size_out + offset_out;

        let mut buf_input = [0.; MAX_CHANNELS];
        I::DECODE_FLOAT_FN(
            if I::REVERSE {
                unimplemented!()
            } else {
                unsafe { &*(&input[pos_in..] as *const _ as *const _) }
            },
            &mut buf_input,
        );

        // TODO: gamut check
        let mut buf_output = [0.; MAX_CHANNELS];
        transform.pipeline.eval_float(&buf_input, &mut buf_output);

        O::ENCODE_FLOAT_FN(
            &buf_output,
            if O::REVERSE {
                unimplemented!()
            } else {
                unsafe { &mut *(&mut output[pos_out..] as *mut _ as *mut _) }
            },
        );
    }
}

/// Handles a transformation using floats.
///
/// The input and output buffers are for all intents and purposes void pointers with lengths.
fn dyn_transform_float(
    transform: &DynTransform,
    in_fmt: &DynPixelFormat,
    out_fmt: &DynPixelFormat,
    input: &[u8],
    output: &mut [u8],
) {
    let px_size_in = in_fmt.size();
    let px_size_out = out_fmt.size();

    if output.len() / px_size_out < input.len() / px_size_in {
        return; // TODO: error
    }

    // TODO: handle extra channels
    // handle_extra_channels(transform, input as *const _ as *const _, output as *mut _ as *mut _, input.len(), 1, 0);

    let offset_in = if in_fmt.extra_first {
        in_fmt.extra_channels * in_fmt.element_size
    } else {
        0
    };
    let offset_out = if out_fmt.extra_first {
        out_fmt.extra_channels * out_fmt.element_size
    } else {
        0
    };

    for i in 0..(input.len() / px_size_in) {
        let pos_in = i * px_size_in + offset_in;
        let pos_out = i * px_size_out + offset_out;

        let mut buf_input = [0.; MAX_CHANNELS];
        (in_fmt.decode_float_fn)(
            in_fmt,
            if in_fmt.reverse {
                unimplemented!()
            } else {
                unsafe { &*(&input[pos_in..] as *const _ as *const _) }
            },
            &mut buf_input,
        );

        // TODO: gamut check
        let mut buf_output = [0.; MAX_CHANNELS];
        transform.pipeline.eval_float(&buf_input, &mut buf_output);

        (out_fmt.encode_float_fn)(
            out_fmt,
            &buf_output,
            if out_fmt.reverse {
                unimplemented!()
            } else {
                unsafe { &mut *(&mut output[pos_out..] as *mut _ as *mut _) }
            },
        );
    }
}
