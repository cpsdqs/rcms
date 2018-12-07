//! Virtual (built-in) profiles

use crate::gamma::ToneCurve;
use crate::mlu::MLU;
use crate::pipe::{Pipeline, Stage};
use crate::profile::Profile;
use crate::white_point::{adaptation_matrix, build_rgb_to_xyz_transfer_matrix, D50};
use crate::{CIExyY, CIExyYTriple, ColorSpace, ICCTag, Intent, ProfileClass, CIEXYZ};

fn set_text_tags(profile: &mut Profile, description: &str) {
    let mut desc_mlu = MLU::new();
    let mut copyright = MLU::new();

    desc_mlu.set("en", "US", description);
    copyright.set("en", "US", "No copyright, use freely");

    profile.insert_tag(ICCTag::ProfileDescription, desc_mlu);
    profile.insert_tag(ICCTag::Copyright, copyright);
}

fn set_seq_desc_tag(_profile: Profile, _model: &str) -> bool {
    // TODO
    unimplemented!()
}

// This function creates a profile based on white point, primaries and
// transfer functions.
fn create_rgb_profile_opt(
    white_point: Option<CIExyY>,
    primaries: Option<CIExyYTriple>,
    transfer_fn: Option<[ToneCurve; 3]>,
) -> Result<Profile, ()> {
    let mut profile = Profile::new(ProfileClass::Display, ColorSpace::RGB);
    profile.set_version(4.3);
    profile.pcs = ColorSpace::XYZ;
    profile.rendering_intent = Intent::Perceptual;

    // Implement profile using following tags:
    //
    //  1 cmsSigProfileDescriptionTag
    //  2 cmsSigMediaWhitePointTag
    //  3 cmsSigRedColorantTag
    //  4 cmsSigGreenColorantTag
    //  5 cmsSigBlueColorantTag
    //  6 cmsSigRedTRCTag
    //  7 cmsSigGreenTRCTag
    //  8 cmsSigBlueTRCTag
    //  9 Chromatic adaptation Tag
    // This conforms a standard RGB DisplayProfile as says ICC, and then I add (As per addendum II)
    // 10 cmsSigChromaticityTag

    set_text_tags(&mut profile, "RGB built-in");

    if let Some(white_point) = white_point {
        profile.insert_tag(ICCTag::MediaWhitePoint, D50);
        let chad = adaptation_matrix(None, white_point.into(), D50).unwrap();

        // this is a v4 tag but many CMS understand and read it regardless
        profile.insert_tag(ICCTag::ChromaticAdaptation, chad);

        if let Some(primaries) = primaries {
            let max_white = CIExyY {
                x: white_point.x,
                y: white_point.y,
                Y: 1.,
            };

            let m_colorants = match build_rgb_to_xyz_transfer_matrix(max_white, primaries) {
                Some(mat) => mat,
                None => return Err(()), // TODO: error
            };

            profile.insert_tag(
                ICCTag::RedColorant,
                CIEXYZ {
                    x: m_colorants[0][0],
                    y: m_colorants[1][0],
                    z: m_colorants[2][0],
                },
            );
            profile.insert_tag(
                ICCTag::GreenColorant,
                CIEXYZ {
                    x: m_colorants[0][1],
                    y: m_colorants[1][1],
                    z: m_colorants[2][1],
                },
            );
            profile.insert_tag(
                ICCTag::BlueColorant,
                CIEXYZ {
                    x: m_colorants[0][2],
                    y: m_colorants[1][2],
                    z: m_colorants[2][2],
                },
            );
        }
    }

    if let Some(transfer_fn) = transfer_fn {
        profile.insert_tag(ICCTag::RedTRC, transfer_fn[0].clone());

        // Tries to minimize space. Thanks to Richard Hughes for this nice idea
        if transfer_fn[1] == transfer_fn[0] {
            profile.link_tag(ICCTag::GreenTRC, ICCTag::RedTRC);
        } else {
            profile.insert_tag(ICCTag::GreenTRC, transfer_fn[1].clone());
        }

        if transfer_fn[2] == transfer_fn[0] {
            profile.link_tag(ICCTag::BlueTRC, ICCTag::RedTRC);
        } else {
            profile.insert_tag(ICCTag::BlueTRC, transfer_fn[2].clone());
        }
    }

    if let Some(primaries) = primaries {
        profile.insert_tag(ICCTag::Chromaticity, primaries);
    }

    Ok(profile)
}

fn create_gray_profile_opt(white_point: Option<CIExyY>, transfer_fn: Option<ToneCurve>) -> Profile {
    let mut profile = Profile::new(ProfileClass::Display, ColorSpace::Gray);

    profile.pcs = ColorSpace::XYZ;
    profile.rendering_intent = Intent::Perceptual;

    // Implement profile using following tags:
    //
    //  1 cmsSigProfileDescriptionTag
    //  2 cmsSigMediaWhitePointTag
    //  3 cmsSigGrayTRCTag

    // This conforms a standard Gray DisplayProfile

    // Fill-in the tags

    set_text_tags(&mut profile, "gray built-in");

    if let Some(white_point) = white_point {
        let wp_xyz: CIEXYZ = white_point.into();
        profile.insert_tag(ICCTag::MediaWhitePoint, wp_xyz);
    }

    if let Some(transfer_fn) = transfer_fn {
        profile.insert_tag(ICCTag::GrayTRC, transfer_fn);
    }

    profile
}

// sRGB Curves are defined by:
//
// If R'sRGB, G'sRGB, B'sRGB < 0.04045
//
//    R =  R'sRGB / 12.92
//    G =  G'sRGB / 12.92
//    B =  B'sRGB / 12.92
//
// else if  R'sRGB,G'sRGB, B'sRGB >= 0.04045
//
//    R = ((R'sRGB + 0.055) / 1.055) ^ 2.4
//    G = ((G'sRGB + 0.055) / 1.055) ^ 2.4
//    B = ((B'sRGB + 0.055) / 1.055) ^ 2.4

fn build_srgb_gamma() -> Result<ToneCurve, ()> {
    ToneCurve::new_parametric(4, &[2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045])
        .map_err(|_| ())
}

/// Creates the ICC virtual profile for the sRGB color space
fn create_srgb_profile_opt() -> Result<Profile, ()> {
    let d65 = CIExyY {
        x: 0.3127,
        y: 0.3290,
        Y: 1.0,
    };

    let rec709_primaries = CIExyYTriple {
        red: CIExyY {
            x: 0.6400,
            y: 0.3300,
            Y: 1.0,
        },
        green: CIExyY {
            x: 0.3000,
            y: 0.6000,
            Y: 1.0,
        },
        blue: CIExyY {
            x: 0.1500,
            y: 0.0600,
            Y: 1.0,
        },
    };

    let gamma22 = [
        build_srgb_gamma()?,
        build_srgb_gamma()?,
        build_srgb_gamma()?,
    ];

    let mut srgb = create_rgb_profile_opt(Some(d65), Some(rec709_primaries), Some(gamma22))?;
    set_text_tags(&mut srgb, "sRGB built-in");
    Ok(srgb)
}

// Creates a fake Lab identity.
pub(crate) fn create_lab2_profile_opt(white_point: Option<CIExyY>) -> Result<Profile, ()> {
    let mut profile = create_rgb_profile_opt(Some(white_point.unwrap_or(D50.into())), None, None)?;

    profile.set_version(2.1);

    profile.device_class = ProfileClass::Abstract;
    profile.color_space = ColorSpace::Lab;
    profile.pcs = ColorSpace::Lab;

    set_text_tags(&mut profile, "Lab identity built-in");

    // An identity LUT is all we need
    let mut lut = Pipeline::new(3, 3);
    lut.prepend_stage(Stage::new_identity(3));
    profile.insert_tag(ICCTag::AToB0, lut);

    Ok(profile)
}

// Creates a fake Lab V4 identity.
pub(crate) fn create_lab4_profile_opt(white_point: Option<CIExyY>) -> Result<Profile, ()> {
    let mut profile = create_rgb_profile_opt(Some(white_point.unwrap_or(D50.into())), None, None)?;

    profile.set_version(4.3);

    profile.device_class = ProfileClass::Abstract;
    profile.color_space = ColorSpace::Lab;
    profile.pcs = ColorSpace::Lab;

    set_text_tags(&mut profile, "Lab identity built-in");

    // An identity LUT is all we need
    let mut lut = Pipeline::new(3, 3);
    lut.prepend_stage(Stage::new_identity_curves(3));
    profile.insert_tag(ICCTag::AToB0, lut);

    Ok(profile)
}

impl Profile {
    /// Creates a new RGB profile.
    pub fn new_rgb(
        white_point: CIExyY,
        primaries: CIExyYTriple,
        transfer_fn: [ToneCurve; 3],
    ) -> Result<Profile, ()> {
        create_rgb_profile_opt(Some(white_point), Some(primaries), Some(transfer_fn))
    }

    /// Creates a new monochrome profile.
    pub fn new_gray(white_point: CIExyY, transfer_fn: ToneCurve) -> Profile {
        create_gray_profile_opt(Some(white_point), Some(transfer_fn))
    }

    /// Creates a new sRGB profile.
    ///
    /// - white point: D65 (x: 0.3127, y: 0.3290, Y: 1.0)
    /// - ITU-R BT.709-5 primaries:
    ///     - R: (x: 0.6400, y: 0.3300, Y: 1.0),
    ///     - G: (x: 0.3000, y: 0.6000, Y: 1.0),
    ///     - B: (x: 0.1500, y: 0.0600, Y: 1.0),
    ///
    /// Transfer functions defined by:
    /// ```python
    /// if R'sRGB, G'sRGB, B'sRGB < 0.04045:
    ///    R = R'sRGB / 12.92
    ///    G = G'sRGB / 12.92
    ///    B = B'sRGB / 12.92
    /// else if R'sRGB,G'sRGB, B'sRGB >= 0.04045:
    ///    R = ((R'sRGB + 0.055) / 1.055) ^ 2.4
    ///    G = ((G'sRGB + 0.055) / 1.055) ^ 2.4
    ///    B = ((B'sRGB + 0.055) / 1.055) ^ 2.4
    /// ```
    pub fn new_srgb() -> Profile {
        // if this panics then something is very wrong
        create_srgb_profile_opt().unwrap()
    }

    /// Creates a new identity Lab v2 profile.
    pub fn new_lab2(wp: CIExyY) -> Result<Profile, ()> {
        create_lab2_profile_opt(Some(wp))
    }

    /// Creates a new identity Lab v4 profile.
    pub fn new_lab4(wp: CIExyY) -> Result<Profile, ()> {
        create_lab4_profile_opt(Some(wp))
    }
}

// TODO: CreateLinearizationDeviceLinkTHR
// TODO: impl Profile CreateLinearizationDeviceLink
// TODO: InkLimitingSampler
// TODO: CreateInkLimitingDeviceLinkTHR
// TODO: impl Profile CreateInkLimitingDeviceLink
// TODO: CreateXYZProfileTHR
// TODO: impl Profile CreateXYZProfile
// TODO: bchswSampler
// TODO: CreateBCHSWabstractProfileTHR
// TODO: impl Profile CreateBCHSWabstractProfile
// TODO: CreateNULLProfileTHR
// TODO: impl Profile CreateNULLProfile
// TODO: IsPCS
// TODO: FixColorSpaces
// TODO: CreateNamedColorDevicelink
// TODO: CheckOne
// TODO: FindCombination
// TODO: cmsTransform2DeviceLink
