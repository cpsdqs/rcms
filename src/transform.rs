use convert::link_profiles;
use lut::Pipeline;
use pixel_format::{PixelFormat, MAX_CHANNELS};
use profile::Profile;
use std::marker::PhantomData;
use std::{fmt, mem, slice};
use transform_tmp::TransformFlags;
use {ColorSpace, Intent, ProfileClass};

type TransformFn<I, O> = fn(transform: &Transform<I, O>, input: &[u8], output: &mut [u8]);

pub struct Transform<InFmt: PixelFormat, OutFmt: PixelFormat> {
    pipeline: Pipeline,
    intent: Intent,
    transform_fn: TransformFn<InFmt, OutFmt>,

    _phantom_in: PhantomData<InFmt>,
    _phantom_out: PhantomData<OutFmt>,
}

impl<InFmt: PixelFormat, OutFmt: PixelFormat> Transform<InFmt, OutFmt> {
    fn alloc(pipeline: Pipeline, intent: Intent) -> Transform<InFmt, OutFmt> {
        let transform_fn = if InFmt::IS_FLOAT & OutFmt::IS_FLOAT {
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

    // TODO: type checks somehow?
    /// Converts values in an input buffer to values in the output buffer.
    pub fn convert<T, U>(&self, input: &[T], output: &mut [U]) {
        let input = unsafe {
            slice::from_raw_parts(
                input.as_ptr() as *const _,
                input.len() * mem::size_of::<T>(),
            )
        };
        let output = unsafe {
            slice::from_raw_parts_mut(
                output.as_ptr() as *mut _,
                output.len() * mem::size_of::<U>(),
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
