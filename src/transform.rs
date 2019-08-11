//! Color space transformations.
//!
//! # Examples
//! ```
//! # use lcms_prime::*;
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
//! let srgb_to_aces = Transform::new(&srgb, &aces_cg, Intent::Perceptual).unwrap();
//!
//! // [R, G, B] pixel
//! let srgb_data = [0.7, 0.3, 0.1];
//! // [R, G, B] pixel
//! let mut aces_cg_data = [0.; 3];
//!
//! // perform the conversion from sRGB to ACEScg
//! srgb_to_aces.convert(&srgb_data, &mut aces_cg_data);
//!
//! println!("sRGB {:?} -> ACEScg {:?}", srgb_data, aces_cg_data);
//! // => sRGB [0.7, 0.3, 0.1] -> ACEScg [0.3000017, 0.0986936, 0.025978323]
//!
//! const TOLERANCE: f64 = 1e-4;
//! fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < TOLERANCE }
//!
//! // test values from ColorSync Utility
//! assert!(approx_eq(aces_cg_data[0], 0.3000));
//! assert!(approx_eq(aces_cg_data[1], 0.0987));
//! assert!(approx_eq(aces_cg_data[2], 0.0260));
//! ```

use crate::convert::link_profiles;
use crate::pipe::Pipeline;
use crate::profile::Profile;
use crate::{ColorSpace, Intent, ProfileClass};
use bitflags::*;

bitflags! {
    pub struct TransformFlags: u32 {
        const NOCACHE =                  0x0040;    // Inhibit 1-pixel cache
        const NOOPTIMIZE =               0x0100;    // Inhibit optimizations
        const NULLTRANSFORM =            0x0200;    // Don't transform anyway

        /// Proofing flags
        const GAMUTCHECK =               0x1000;    // Out of Gamut alarm
        const SOFTPROOFING =             0x4000;    // Do softproofing

        /// Misc
        const BLACKPOINTCOMPENSATION =   0x2000;
        const NOWHITEONWHITEFIXUP =      0x0004;    // Don't fix scum dot
        const HIGHRESPRECALC =           0x0400;    // Use more memory to give better accurancy
        const LOWRESPRECALC =            0x0800;    // Use less memory to minimize resources

        /// For devicelink creation
        const F8BITS_DEVICELINK =         0x0008;   // Create 8 bits devicelinks
        const GUESSDEVICECLASS =         0x0020;   // Guess device class (for transform2devicelink)
        const KEEP_SEQUENCE =            0x0080;   // Keep profile sequence for devicelink creation

        /// Specific to a particular optimizations
        const FORCE_CLUT =               0x0002;    // Force CLUT optimization
        const CLUT_POST_LINEARIZATION =  0x0001;    // create postlinearization tables if possible
        const CLUT_PRE_LINEARIZATION =   0x0010;    // create prelinearization tables if possible

        /// Specific to unbounded mode
        const NONEGATIVES =              0x8000;    // Prevent negative numbers in floating point transforms

        /// Copy alpha channels when transforming
        const COPY_ALPHA =               0x04000000; // Alpha channels are copied on cmsDoTransform()

        /// Internal
        const __CAN_CHANGE_FORMATTER =     0x02000000;
    }
}

/// A color transform.
#[derive(Debug, Clone)]
pub struct Transform {
    pipeline: Pipeline,
    intent: Intent,
}

impl Transform {
    /// Creates a new transform with two profiles.
    pub fn new(
        input: &Profile,
        output: &Profile,
        intent: Intent,
    ) -> Result<Transform, String> {
        let profiles = vec![input.clone(), output.clone()];

        Self::new_multi(&profiles, intent, false)
    }

    /// Creates a new transform with arbitrarily many profiles, an intent, and optional black point compensation.
    pub fn new_multi(
        profiles: &[Profile],
        intent: Intent,
        bpc: bool,
    ) -> Result<Transform, String> {
        let mut bpcs = Vec::with_capacity(256);
        let mut intents = Vec::with_capacity(256);
        let mut adaptation_states = Vec::with_capacity(256);

        for _ in profiles {
            // bpc.push(dw_flags.contains(TransformFlags::BLACKPOINTCOMPENSATION));
            bpcs.push(bpc);
            intents.push(intent);
            adaptation_states.push(-1.); // TODO: check if this is fine
        }

        Self::new_ext(
            profiles,
            &bpcs,
            &intents,
            &adaptation_states,
        )
    }

    /// Creates a new transform.
    ///
    /// profiles, bpc (black point compensation), intents, and adaptation states must all have the
    /// same length.
    pub fn new_ext(
        profiles: &[Profile],
        bpc: &[bool],
        intents: &[Intent],
        adaptation_states: &[f64],
    ) -> Result<Transform, String> {
        // TODO: better errors
        let (entry_cs, exit_cs) = match input_output_spaces(profiles) {
            Some(x) => x,
            None => return Err("Too few profiles in transform".into()),
        };

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

        let transform = Transform {
            pipeline,
            intent: *intents.last().unwrap()
        };

        // TODO: gamut check
        // TODO: 16-bit colorant stuff
        // TODO: cache stuff etc.

        Ok(transform)
    }

    /// Converts values from the input color space to the output color space.
    ///
    /// Care should be taken to ensure the pixel formats are correct.
    pub fn convert(&self, input: &[f64], output: &mut [f64]) {
        self.pipeline.eval_float(input, output);
    }

    /// Returns the inner pipeline.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
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
