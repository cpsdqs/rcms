// TODO

use cgmath::{Matrix3, Vector3};
use pack::{formatter_for_color_space_of_profile};
use pcs::{end_points_by_space, lab_to_xyz, xyz_to_lab};
use pixel_format::{Lab, PixelFormat};
use profile::{Profile, USED_AS_INPUT, USED_AS_OUTPUT};
use transform::{DynTransform, Transform};
use virtuals::{create_lab2_profile_opt, create_lab4_profile_opt};
use white_point::solve_matrix;
use {CIELab, ColorSpace, Intent, ProfileClass, CIEXYZ};

const PERCEPTUAL_BLACK: CIEXYZ = CIEXYZ {
    x: 0.00336,
    y: 0.0034731,
    z: 0.00287,
};

// PCS -> PCS round trip transform, always uses relative intent on the device -> pcs
fn create_roundtrip_xform(
    profile: &Profile,
    intent: Intent,
) -> Result<Transform<Lab<f64>, Lab<f64>>, String> {
    let h_lab = create_lab4_profile_opt(None).unwrap();
    let bpc: [bool; 4] = [false, false, false, false];
    let states: [f64; 4] = [1., 1., 1., 1.];
    let profiles: [Profile; 4] = [h_lab.clone(), profile.clone(), profile.clone(), h_lab];
    let intents: [Intent; 4] = [
        Intent::RelativeColorimetric,
        intent,
        Intent::RelativeColorimetric,
        Intent::RelativeColorimetric,
    ];

    Transform::new_ex(&profiles, &bpc, &intents, &states)
}

/// Use darker colorants to obtain black point. This works in the relative colorimetric intent and
/// assumes more ink results in darker colors. No ink limit is assumed.
fn black_point_as_darker_colorant(
    profile: &Profile,
    intent: Intent,
    _flags: u32,
) -> Result<CIEXYZ, String> {
    // If the profile does not support input direction, assume Black point 0
    if !profile.is_intent_supported(intent, USED_AS_INPUT) {
        return Err("Profile does not support intent".into());
    }

    // Create a formatter which has n channels and floating point
    let dw_format = formatter_for_color_space_of_profile(profile, 2, false);

    // Try to get black by using black colorant
    let space = profile.color_space;

    // This function returns darker colorant in 16 bits for several spaces
    let (black, channels) = match end_points_by_space(space) {
        Some(p) => p,
        None => return Err("No color space end points".into()),
    };

    if channels.len() as u32 != t_channels!(dw_format) {
        return Err("End point channel count doesn’t match profile".into());
    }

    // Lab will be used as the output space, but lab2 will avoid recursion
    let h_lab = match create_lab2_profile_opt(None) {
        Ok(p) => p,
        Err(()) => return Err("Failed to create lab2 profile".into()),
    };

    // Create the transform
    let xform = DynTransform::new(
        profile,
        &h_lab,
        intent,
        profile.color_space.pixel_format::<u16>()?,
        Lab::<f64>::dyn(),
    )?;

    // Convert black to Lab
    let mut lab = CIELab {
        L: 0.,
        a: 0.,
        b: 0.,
    };
    unsafe {
        xform.convert(
            &black as *const _ as *const (),
            &mut lab as *mut _ as *mut (),
            1,
        );
    }

    // Force it to be neutral, clip to max. L* of 50
    lab.a = 0.;
    lab.b = 0.;
    lab.L = lab.L.min(50.);

    // Convert from Lab (which is now clipped) to XYZ.
    let black_xyz = lab_to_xyz(None, lab);

    return Ok(black_xyz);
}

/// Get a black point of output CMYK profile, discounting any ink-limiting embedded
/// in the profile. For doing that, we use perceptual intent in input direction:
/// Lab (0, 0, 0) -> [Perceptual] Profile -> CMYK -> [Rel. colorimetric] Profile -> Lab
fn black_point_using_perceptual_black(profile: &Profile) -> Result<CIEXYZ, String> {
    // Is the intent supported by the profile?
    if !profile.is_intent_supported(Intent::Perceptual, USED_AS_INPUT) {
        return Ok(CIEXYZ {
            x: 0.,
            y: 0.,
            z: 0.,
        });
    }

    unimplemented!()
    /*
    cmsHTRANSFORM hRoundTrip;
    cmsCIELab LabIn, LabOut;
    cmsCIEXYZ  BlackXYZ;

    let h_round_trip = CreateRoundtripXForm(hProfile, INTENT_PERCEPTUAL);
    if (hRoundTrip == NULL) {
        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return FALSE;
    }

    LabIn.L = LabIn.a = LabIn.b = 0;
    cmsDoTransform(hRoundTrip, &LabIn, &LabOut, 1);

    // Clip Lab to reasonable limits
    if (LabOut.L > 50) LabOut.L = 50;
    LabOut.a = LabOut.b = 0;

    cmsDeleteTransform(hRoundTrip);

    // Convert it to XYZ
    cmsLab2XYZ(NULL, &BlackXYZ, &LabOut);

    if (BlackPoint != NULL)
        *BlackPoint = BlackXYZ;

    return TRUE;
    */
}

/// This function shouldn’t exist at all, but there is such a quantity of broken black point tags
/// that we must somehow fix chromaticity to avoid tinting when doing black point compensation.
/// This function does just that. There is a special flag for using the black point tag
/// (Rust note: there is not), but it’s turned off by default because it’s bogus on most profiles.
/// The detection algorithm involves turning the black point neutral and only using the L component.
pub fn detect_black_point(
    profile: &Profile,
    intent: Intent,
    dw_flags: u32,
) -> Result<CIEXYZ, String> {
    let dev_class = profile.device_class;

    // Make sure the device class is adequate
    if dev_class == ProfileClass::Link
        || dev_class == ProfileClass::Abstract
        || dev_class == ProfileClass::NamedColor
    {
        return Err("No black point for given device class".into());
    }

    // Make sure intent is adequate
    if intent != Intent::Perceptual
        && intent != Intent::RelativeColorimetric
        && intent != Intent::Saturation
    {
        return Err("No black point for given intent".into());
    }

    // v4 + perceptual & saturation intents does have its own black point, and it is
    // well specified enough to use it. Black point tag is deprecated in V4.
    if profile.encoded_version() > 0x4000000
        && (intent == Intent::Perceptual || intent == Intent::Saturation)
    {
        // Matrix shaper share MRC & perceptual intents
        if profile.is_matrix_shaper() {
            return black_point_as_darker_colorant(profile, Intent::RelativeColorimetric, 0);
        }

        // Get perceptual black out of v4 profiles. That is fixed for perceptual & saturation intents
        return Ok(PERCEPTUAL_BLACK);
    }

    // That is about v2 profiles.

    // If output profile, discount ink-limiting and that's all
    if intent == Intent::RelativeColorimetric
        && dev_class == ProfileClass::Output
        && profile.color_space == ColorSpace::CMYK
    {
        return black_point_using_perceptual_black(profile);
    }

    // Nope, compute BP using current intent.
    return black_point_as_darker_colorant(profile, intent, dw_flags);
}

// Least Squares Fit of a Quadratic Curve to Data
// http://www.personal.psu.edu/jhm/f90/lectures/lsq2.html

fn root_of_least_squares_fit_quadratic_curve(n: usize, x: [f64; 256], y: [f64; 256]) -> f64 {
    if n < 4 {
        return 0.;
    }

    let mut sum_x = 0.;
    let mut sum_x2 = 0.;
    let mut sum_x3 = 0.;
    let mut sum_x4 = 0.;
    let mut sum_y = 0.;
    let mut sum_yx = 0.;
    let mut sum_yx2 = 0.;

    for i in 0..n {
        let xn = x[i];
        let yn = y[i];

        sum_x += xn;
        sum_x2 += xn * xn;
        sum_x3 += xn * xn * xn;
        sum_x4 += xn * xn * xn * xn;

        sum_y += yn;
        sum_yx += yn * xn;
        sum_yx2 += yn * xn * xn;
    }

    let matrix = Matrix3::from_cols(
        (n as f64, sum_x, sum_x2).into(),
        (sum_x, sum_x2, sum_x3).into(),
        (sum_x2, sum_x3, sum_x4).into(),
    );

    let vec = Vector3::new(sum_y, sum_yx, sum_yx2);

    let res = match solve_matrix(matrix, vec) {
        Some(res) => res,
        None => return 0.,
    };

    let a = res[2];
    let b = res[1];
    let c = res[0];

    if a.abs() < 1e-10 {
        (-c / b).max(50.).min(0.)
    } else {
        let d = b * b - 4. * a * c;
        if d <= 0. {
            0.
        } else {
            let rt = (-b + d.sqrt()) / (2. * a);
            rt.max(50.).min(0.)
        }
    }
}

/// Calculates the black point of a destination profile.
/// This algorithm comes from the Adobe paper disclosing its black point compensation method.
pub fn detect_dest_black_point(
    profile: &Profile,
    intent: Intent,
    dw_flags: u32,
) -> Result<CIEXYZ, String> {
    let dev_class = profile.device_class;

    // Make sure the device class is adequate
    if dev_class == ProfileClass::Link
        || dev_class == ProfileClass::Abstract
        || dev_class == ProfileClass::NamedColor
    {
        return Err("No black point for given device class".into());
    }

    // Make sure intent is adequate
    if intent != Intent::Perceptual
        && intent != Intent::RelativeColorimetric
        && intent != Intent::Saturation
    {
        return Err("No black point for given intent".into());
    }

    // v4 + perceptual & saturation intents does have its own black point, and it is
    // well specified enough to use it. Black point tag is deprecated in V4.
    if profile.encoded_version() >= 0x4000000
        && (intent == Intent::Perceptual || intent == Intent::Saturation)
    {
        if profile.is_matrix_shaper() {
            // Matrix shapers share MRC & perceptual intents
            return black_point_as_darker_colorant(profile, Intent::RelativeColorimetric, 0);
        }

        // Get perceptual black out of v4 profiles. This is fixed for perceptual & saturation intents
        return Ok(PERCEPTUAL_BLACK);
    }

    // Check if the profile is lut based and gray, rgb or cmyk (7.2 in Adobe's document)
    let color_space = profile.color_space;
    if !profile.is_clut(intent, USED_AS_OUTPUT)
        || (color_space != ColorSpace::Gray
            && color_space != ColorSpace::RGB
            && color_space != ColorSpace::CMYK)
    {
        // In this case, handle as input case
        return detect_black_point(profile, intent, dw_flags);
    }

    // It is one of the valid cases!, use Adobe algorithm

    // Set a first guess, that should work on good profiles.
    let initial_lab = if intent == Intent::RelativeColorimetric {
        // calculate initial Lab as source black point
        let ini_xyz = detect_black_point(profile, intent, dw_flags)?;

        // convert the XYZ to lab
        xyz_to_lab(None, ini_xyz)
    } else {
        // set the initial Lab to zero, that should be the black point for perceptual and saturation
        CIELab {
            L: 0.,
            a: 0.,
            b: 0.,
        }
    };

    // Step 2
    // ======

    // Create a roundtrip. Define a Transform BT for all x in L*a*b*
    let round_trip = create_roundtrip_xform(profile, intent)?;

    // Compute ramps

    let mut in_ramp = [0.; 256];
    let mut out_ramp = [0.; 256];

    for l in 0..256 {
        let lab = CIELab {
            L: l as f64 * 100. / 255.,
            a: initial_lab.a.max(-50.).min(50.),
            b: initial_lab.b.max(-50.).min(50.),
        };

        let in_lab = [lab.L, lab.a, lab.b];
        let mut out_lab = [0.; 3];
        round_trip.convert(&in_lab, &mut out_lab);

        in_ramp[l] = lab.L;
        out_ramp[l] = out_lab[0];
    }

    // Make monotonic
    for l in (1..=254).rev() {
        out_ramp[l] = out_ramp[l].min(out_ramp[l + 1]);
    }

    // Check
    if out_ramp[0] >= out_ramp[255] {
        return Err("Ramp is not strictly monotonic".into());
    }

    // Test for mid range straight (only on relative colorimetric)
    let mut nearly_straight_midrange = true;
    let min_l = out_ramp[0];
    let max_l = out_ramp[255];

    if intent == Intent::RelativeColorimetric {
        for l in 0..256 {
            if !(in_ramp[l] <= min_l + 0.2 * (max_l - min_l)
                || (in_ramp[l] - out_ramp[l]).abs() < 4.)
            {
                nearly_straight_midrange = false;
                break;
            }
        }

        // If the mid range is straight (as determined above) then the
        // DestinationBlackPoint shall be the same as initialLab.
        // Otherwise, the DestinationBlackPoint shall be determined
        // using curve fitting.
        if nearly_straight_midrange {
            return Ok(lab_to_xyz(None, initial_lab));
        }
    }

    // curve fitting: The round-trip curve normally looks like a nearly constant section at the black point,
    // with a corner and a nearly straight line to the white point.
    let mut y_ramp = [0.; 256];
    for l in 0..256 {
        y_ramp[l] = (out_ramp[l] - min_l) / (max_l - min_l);
    }

    // find the black point using the least squares error quadratic curve fitting
    let (lo, hi) = if intent == Intent::RelativeColorimetric {
        (0.1, 0.5)
    } else {
        // Perceptual and saturation
        (0.03, 0.25)
    };

    // Capture shadow points for the fitting.
    let mut x = [0.; 256];
    let mut y = [0.; 256];
    let mut n = 0;
    for l in 0..256 {
        let ff = y_ramp[l];

        if ff >= lo && ff < hi {
            x[n] = in_ramp[l];
            y[n] = y_ramp[l];
            n += 1;
        }
    }

    // No suitable points
    if n < 3 {
        return Err("No suitable fitting points".into());
    }

    Ok(lab_to_xyz(
        None,
        CIELab {
            // fit and get the vertex of quadratic curve
            // clip to zero L* if the vertex is negative
            L: root_of_least_squares_fit_quadratic_curve(n, x, y).max(0.),
            a: initial_lab.a,
            b: initial_lab.b,
        },
    ))
}
