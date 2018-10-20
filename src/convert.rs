//! Color conversion.

use cgmath::{Matrix3, SquareMatrix, Vector3, Zero};
use pcs::MAX_ENCODABLE_XYZ;
use pipe::{Pipeline, Stage};
use profile::Profile;
use sampling::{detect_black_point, detect_dest_black_point};
use transform::TransformFlags;
use white_point::{
    adaptation_matrix, mat3_eval, mat3_per, temp_from_white_point, white_point_from_temp, D50,
};
use {ColorSpace, Intent, ProfileClass, CIEXYZ};

type IntentFn = fn(
    profiles: &[Profile],
    intents: &[Intent],
    bpc: &[bool],
    adaptation_states: &[f64],
    flags: TransformFlags,
) -> Result<Pipeline, String>;

#[derive(Clone)]
struct IntentsListItem {
    intent: Intent,
    description: &'static str,
    link: IntentFn,
}

static DEFAULT_INTENTS: [IntentsListItem; 10] = [
    IntentsListItem {
        intent: Intent::Perceptual,
        description: "Perceptual",
        link: default_icc_intents,
    },
    IntentsListItem {
        intent: Intent::RelativeColorimetric,
        description: "Relative colorimetric",
        link: default_icc_intents,
    },
    IntentsListItem {
        intent: Intent::Saturation,
        description: "Saturation",
        link: default_icc_intents,
    },
    IntentsListItem {
        intent: Intent::AbsoluteColorimetric,
        description: "Absolute colorimetric",
        link: default_icc_intents,
    },
    IntentsListItem {
        intent: Intent::PreserveKOnlyPerceptual,
        description: "Perceptual preserving black ink",
        link: black_preserving_k_only_intents,
    },
    IntentsListItem {
        intent: Intent::PreserveKOnlyRelativeColorimetric,
        description: "Relative colorimetric preserving black ink",
        link: black_preserving_k_only_intents,
    },
    IntentsListItem {
        intent: Intent::PreserveKOnlySaturation,
        description: "Saturation preserving black ink",
        link: black_preserving_k_only_intents,
    },
    IntentsListItem {
        intent: Intent::PreserveKPlanePerceptual,
        description: "Perceptual preserving black plane",
        link: black_preserving_k_plane_intents,
    },
    IntentsListItem {
        intent: Intent::PreserveKPlaneRelativeColorimetric,
        description: "Relative colorimetric preserving black plane",
        link: black_preserving_k_plane_intents,
    },
    IntentsListItem {
        intent: Intent::PreserveKPlaneSaturation,
        description: "Saturation preserving black plane",
        link: black_preserving_k_plane_intents,
    },
];

fn search_intent(key: Intent) -> Option<&'static IntentsListItem> {
    for intent in &DEFAULT_INTENTS {
        if intent.intent == key {
            return Some(intent);
        }
    }
    None
}

/// Black point compensation. Implemented as a linear scaling in XYZ. Black points
/// should come relative to the white point. Fills a matrix/offset element m
/// which is organized as a 4x4 matrix.
fn compute_black_point_compensation(
    black_point_in: CIEXYZ,
    black_point_out: CIEXYZ,
) -> (Matrix3<f64>, Vector3<f64>) {
    // Now we need to compute a matrix plus an offset m such that
    // [m]*bpin + off = bpout
    // [m]*D50  + off = D50
    //
    // This is a linear scaling in the form ax+b, where
    // a = (bpout - D50) / (bpin - D50)
    // b = - D50* (bpout - bpin) / (bpin - D50)
    let tx = black_point_in.x - D50.x;
    let ty = black_point_in.y - D50.y;
    let tz = black_point_in.z - D50.z;

    let ax = (black_point_out.x - D50.x) / tx;
    let ay = (black_point_out.y - D50.y) / ty;
    let az = (black_point_out.z - D50.z) / tz;

    let bx = -D50.x * (black_point_out.x - black_point_in.x) / tx;
    let by = -D50.y * (black_point_out.y - black_point_in.y) / ty;
    let bz = -D50.z * (black_point_out.z - black_point_in.z) / tz;

    (
        Matrix3::from_diagonal((ax, ay, az).into()),
        Vector3::new(bx, by, bz),
    )
}

/// Approximate a blackbody illuminant based on CHAD information
fn chad_to_temp(chad: Matrix3<f64>) -> Option<f64> {
    let inverse = match chad.invert() {
        Some(inverse) => inverse,
        None => return None,
    };

    let d50_xyz = D50;
    let s = Vector3::new(d50_xyz.x, d50_xyz.y, d50_xyz.z);
    let d = mat3_eval(inverse, s);

    let dest = CIEXYZ {
        x: d.x,
        y: d.y,
        z: d.z,
    };

    temp_from_white_point(dest.into())
}

/// Compute a CHAD based on a given temperature
fn temp_to_chad(temp: f64) -> Option<Matrix3<f64>> {
    let chromaticity_of_white = match white_point_from_temp(temp) {
        Some(c) => c,
        None => return None,
    };
    adaptation_matrix(None, chromaticity_of_white.into(), D50)
}

/// Join scalings to obtain relative input to absolute and then to relative output.
/// Result is stored in a 3x3 matrix
fn compute_absolute_intent(
    adaptation_state: f64,
    wp_in: CIEXYZ,
    cam_in: Matrix3<f64>,
    wp_out: CIEXYZ,
    cam_out: Matrix3<f64>,
) -> Result<Matrix3<f64>, String> {
    if adaptation_state == 1. {
        // Observer is fully adapted. Keep chromatic adaptation.
        // That is the standard V4 behaviour
        Ok(Matrix3::from_diagonal(
            (wp_in.x / wp_out.x, wp_in.y / wp_out.y, wp_in.z / wp_out.z).into(),
        ))
    } else {
        // Incomplete adaptation. This is an advanced feature.
        let scale = Matrix3::from_diagonal(
            (wp_in.x / wp_out.x, wp_in.y / wp_out.y, wp_in.z / wp_out.z).into(),
        );

        if adaptation_state == 0. {
            let m2 = mat3_per(cam_out, scale);
            // m2 holds CHAD from output white to D50 times abs. col. scaling

            // Observer is not adapted, undo the chromatic adaptation
            // let m = mat3_per(m2, cam_out);

            let cam_in_inv = match cam_in.invert() {
                Some(m) => m,
                None => return Err("Could not invert CHAD".into()),
            };

            Ok(mat3_per(m2, cam_in_inv))
        } else {
            let m2 = match cam_in.invert() {
                Some(m) => m,
                None => return Err("Could not invert CHAD".into()),
            };

            let m3 = mat3_per(m2, scale);

            // m3 holds CHAD from input white to D50 times abs. col. scaling
            let temp_src = match chad_to_temp(cam_in) {
                Some(v) => v,
                None => return Err("Could not convert CHAD to temperature".into()),
            };
            let temp_dest = match chad_to_temp(cam_out) {
                Some(v) => v,
                None => return Err("Could not convert CHAD to temperature".into()),
            };

            if temp_src < 0. || temp_dest < 0. {
                // something went wrong
                return Err("CHAD returned negative temperature".into());
            }

            if scale.is_identity() && (temp_src - temp_dest).abs() < 0.01 {
                return Ok(Matrix3::<f64>::identity());
            }

            let temp = (1. - adaptation_state) * temp_dest + adaptation_state * temp_src;

            let mixed_chad = match temp_to_chad(temp) {
                Some(m) => m,
                None => return Err("Could not convert temperature to CHAD".into()),
            };

            Ok(mat3_per(m3, mixed_chad))
        }
    }
}

/// Returns if m should be applied
fn is_empty_layer(m: Matrix3<f64>, off: Vector3<f64>) -> bool {
    let ident = Matrix3::<f64>::identity();
    let mut diff = 0.;

    for i in 0..9 {
        diff += (m[i / 3][i % 3] - ident[i / 3][i % 3]).abs();
    }

    for i in 0..3 {
        diff += off[i].abs();
    }

    diff < 0.002
}

/// Compute the conversion layer
fn compute_conversion(
    i: u32,
    profiles: &[Profile],
    intent: Intent,
    bpc: bool,
    adaptation_state: f64,
) -> Result<(Matrix3<f64>, Vector3<f64>), String> {
    let i = i as usize;
    if intent == Intent::AbsoluteColorimetric {
        let prev_profile = &profiles[i - 1];
        let profile = &profiles[i];

        let wp_in = prev_profile.media_white_point();
        let cam_in = prev_profile.chad();
        let wp_out = profile.media_white_point();
        let cam_out = profile.chad();

        Ok((
            compute_absolute_intent(adaptation_state, wp_in, cam_in, wp_out, cam_out)?,
            Vector3::zero(),
        ))
    } else if bpc {
        let bp = detect_black_point(&profiles[i - 1], intent, 0);
        let dest_bp = detect_dest_black_point(&profiles[i], intent, 0);
        if let (Ok(bp_in), Ok(bp_out)) = (bp, dest_bp) {
            // If black points are equal, then do nothing
            if bp_in != bp_out {
                let (m, off) = compute_black_point_compensation(bp_in, bp_out);
                // Offset should be adjusted because the encoding. We encode XYZ normalized to 0..1.0,
                // to do that, we divide by MAX_ENCODEABLE_XYZ. The conversion stage goes XYZ -> XYZ so
                // we have first to convert from encoded to XYZ and then convert back to encoded.
                // y = Mx + Off
                // x = x'c
                // y = M x'c + Off
                // y = y'c; y' = y / c
                // y' = (Mx'c + Off) /c = Mx' + (Off / c)
                Ok((m, off / MAX_ENCODABLE_XYZ))
            } else {
                Ok((Matrix3::identity(), Vector3::zero()))
            }
        } else {
            Ok((Matrix3::identity(), Vector3::zero()))
        }
    } else {
        Ok((Matrix3::identity(), Vector3::zero()))
    }
}

fn add_conversion(
    result: &mut Pipeline,
    in_pcs: ColorSpace,
    out_pcs: ColorSpace,
    mat: Matrix3<f64>,
    off: Vector3<f64>,
) -> Result<(), String> {
    let mat_as_dbl = unsafe { &*(&mat as *const _ as *const [f64; 9]) };
    let off_as_dbl = unsafe { &*(&off as *const _ as *const [f64; 3]) };

    // Handle PCS mismatches. A specialized stage is added to the LUT in such case
    match (in_pcs, out_pcs) {
        (ColorSpace::XYZ, ColorSpace::XYZ) => if !is_empty_layer(mat, off) {
            result.append_stage(Stage::new_matrix(3, 3, mat_as_dbl, Some(off_as_dbl)));
        },
        (ColorSpace::XYZ, ColorSpace::Lab) => {
            if !is_empty_layer(mat, off) {
                result.append_stage(Stage::new_matrix(3, 3, mat_as_dbl, Some(off_as_dbl)));
            }
            result.append_stage(Stage::new_xyz_to_lab());
        }
        (ColorSpace::Lab, ColorSpace::XYZ) => {
            result.append_stage(Stage::new_lab_to_xyz());
            if !is_empty_layer(mat, off) {
                result.append_stage(Stage::new_matrix(3, 3, mat_as_dbl, Some(off_as_dbl)));
            }
        }
        (ColorSpace::Lab, ColorSpace::Lab) => if !is_empty_layer(mat, off) {
            result.append_stage(Stage::new_lab_to_xyz());
            result.append_stage(Stage::new_matrix(3, 3, mat_as_dbl, Some(off_as_dbl)));
            result.append_stage(Stage::new_xyz_to_lab());
        },
        _ => {
            // On colorspaces other than PCS, check for same space
            if in_pcs != out_pcs {
                return Err("PCS don’t match".into());
            }
        }
    }

    Ok(())
}

fn color_space_is_compatible(a: ColorSpace, b: ColorSpace) -> bool {
    if a == b {
        // they're the same
        true
    } else if a == ColorSpace::S4Color && b == ColorSpace::CMYK
        || a == ColorSpace::CMYK && b == ColorSpace::S4Color
    {
        // mch4 substitution of cmyk
        true
    } else if a == ColorSpace::XYZ && b == ColorSpace::Lab
        || a == ColorSpace::Lab && b == ColorSpace::XYZ
    {
        // xyz/lab
        true
    } else {
        false
    }
}

/// This is the default routine for ICC-style intents. A user may decide to override it by using a plugin.
/// Supported intents are perceptual, relative colorimetric, saturation and ICC-absolute colorimetric.
fn default_icc_intents(
    profiles: &[Profile],
    intents: &[Intent],
    bpc: &[bool],
    adaptation_states: &[f64],
    flags: TransformFlags,
) -> Result<Pipeline, String> {
    // For safety
    if profiles.is_empty() {
        return Err("No profiles".into());
    }

    // Allocate an empty LUT for holding the result. 0 as channel count means 'undefined'
    let mut result = Pipeline::new(0, 0);

    let mut current_cs = profiles[0].color_space;

    for i in 0..profiles.len() {
        let profile = &profiles[i];
        let is_device_link = profile.device_class == ProfileClass::Link
            || profile.device_class == ProfileClass::Abstract;

        let is_input = if i == 0 && !is_device_link {
            // First profile is used as input unless devicelink or abstract
            true
        } else {
            // Else use profile in the input direction if current space is not PCS
            current_cs != ColorSpace::XYZ && current_cs != ColorSpace::Lab
        };

        let intent = intents[i];

        let (cs_in, cs_out) = if is_input || is_device_link {
            (profile.color_space, profile.pcs)
        } else {
            (profile.pcs, profile.color_space)
        };

        if !color_space_is_compatible(cs_in, current_cs) {
            return Err("Color space mismatch".into());
        }

        let mut pipeline;

        // If devicelink is found, then no custom intent is allowed and we can
        // read the LUT to be applied. Settings don't apply here.
        if is_device_link
            || (profile.device_class == ProfileClass::NamedColor && profiles.len() == 1)
        {
            // Get the involved LUT from the profile
            pipeline = profile.read_devicelink_lut(intent)?;

            // What about abstract profiles?
            let (m, off) = if profile.device_class == ProfileClass::Abstract && i > 0 {
                compute_conversion(i as u32, profiles, intent, bpc[i], adaptation_states[i])?
            } else {
                (Matrix3::identity(), Vector3::zero())
            };

            add_conversion(&mut result, current_cs, cs_in, m, off)?;
        } else {
            if is_input {
                // Input direction means non-pcs connection, so proceed like devicelinks
                pipeline = profile.read_input_lut(intent)?;
            } else {
                pipeline = profile.read_output_lut(intent)?;

                let (m, off) =
                    compute_conversion(i as u32, profiles, intent, bpc[i], adaptation_states[i])?;
                add_conversion(&mut result, current_cs, cs_in, m, off)?;
            }
        }

        result.concat(&pipeline);

        // Update current space
        current_cs = cs_out;
    }

    // Check for non-negatives clip
    if flags.contains(TransformFlags::NONEGATIVES) {
        if current_cs == ColorSpace::Gray
            || current_cs == ColorSpace::RGB
            || current_cs == ColorSpace::CMYK
        {
            let clip = Stage::new_clip_negatives(current_cs.channels());
            result.append_stage(clip);
        }
    }

    Ok(result)
}

/// Translate black-preserving intents to ICC ones
fn translate_non_icc_intents(intent: Intent) -> Intent {
    match intent {
        Intent::PreserveKOnlyPerceptual | Intent::PreserveKPlanePerceptual => Intent::Perceptual,
        Intent::PreserveKOnlyRelativeColorimetric | Intent::PreserveKPlaneRelativeColorimetric => {
            Intent::RelativeColorimetric
        }
        Intent::PreserveKOnlySaturation | Intent::PreserveKPlaneSaturation => Intent::Saturation,
        _ => intent,
    }
}

/// Preserve black only if that is the only ink used
// fn black_preserving_gray_only_sampler(in: &[u16], out: &[u16], )
// TODO

// This is the entry for black-preserving K-only intents, which are non-ICC. The last profile must be an output profile
// to do the trick (no devicelinks allowed at that position)
fn black_preserving_k_only_intents(
    _profiles: &[Profile],
    _intents: &[Intent],
    _bpc: &[bool],
    _adaptation_states: &[f64],
    _dw_flags: TransformFlags,
) -> Result<Pipeline, String> {
    unimplemented!()
}

// fn black_preserving_sampler

// This is the entry for black-plane preserving, which are non-ICC. The last profile must be an output profile
// to do the trick (no devicelinks allowed at that position)
fn black_preserving_k_plane_intents(
    _profiles: &[Profile],
    _intents: &[Intent],
    _bpc: &[bool],
    _adaptation_states: &[f64],
    _dw_flags: TransformFlags,
) -> Result<Pipeline, String> {
    unimplemented!()
}

/// Link several profiles to obtain a single LUT modelling the whole color transform. Intents, Black point
/// compensation and Adaptation parameters may vary across profiles. BPC and Adaptation refers to the PCS
/// after the profile, i.e. BPC[0] refers to connexion between profile(0) and profile(1)
pub(super) fn link_profiles(
    profiles: &[Profile],
    intents: &[Intent],
    bpc: &[bool],
    adaptation_states: &[f64],
    flags: TransformFlags,
) -> Result<Pipeline, String> {
    let mut bpc = bpc.to_owned();

    for i in 0..profiles.len() {
        // Check if black point is really needed or allowed. Note that
        // following Adobe's document:
        // BPC does not apply to devicelink profiles, nor to abs colorimetric,
        // and applies always on V4 perceptual and saturation.
        if intents[i] == Intent::AbsoluteColorimetric {
            bpc[i] = false;
        }

        if intents[i] == Intent::Perceptual || intents[i] == Intent::Saturation {
            if profiles[i].version() > 4. {
                bpc[i] = true;
            }
        }
    }

    match search_intent(intents[0]) {
        Some(intent) => (intent.link)(profiles, intents, &bpc, adaptation_states, flags),
        None => Err("No such intent".into()),
    }
}
