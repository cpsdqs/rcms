//! Inter-PCS conversions (XYZ <-> CIE L* a* b*)
//!
//! CIE 15:2004 CIELab is defined as:
//! ```text
//! L* = 116*f(Y/Yn) - 16         | 0 <= L* <= 100
//! a* = 500*[f(X/Xn) - f(Y/Yn)]
//! b* = 200*[f(Y/Yn) - f(Z/Zn)]
//! ```
//! and
//! ```text
//! f(t) = t^(1/3)                | 1 >= t >  (24/116)^3
//!        (841/108)*t + (16/116) | 0 <= t <= (24/116)^3
//! ```
//!
//! Reverse transform is:
//! ```text
//! X = Xn*[a* / 500 + (L* + 16) / 116] ^ 3 | if (X/Xn) > (24/116)
//!   = Xn*(a* / 500 + L* / 116) / 7.787    | if (X/Xn) <= (24/116)
//! ```
//!
//! PCS in Lab2 is encoded as:
//!
//! 8 bit Lab PCS:
//! ```text
//! L*  0..100 into a 0..ff byte.
//! a*  t + 128 range is -128.0  +127.0
//! b*
//! ```
//! 16 bit Lab PCS:
//! ```text
//! L*  0..100  into a 0..ff00 word.
//! a*  t + 128  range is  -128.0  +127.9961
//! b*
//! ```
//!
//! | Interchange Space  | Component | Actual Range       | Encoded Range              |
//! |:------------------ |:--------- |:------------------ |:-------------------------- |
//! | CIE XYZ            | X         | 0 -> 1.99997       | 0x0000 -> 0xffff           |
//! | CIE XYZ            | Y         | 0 -> 1.99997       | 0x0000 -> 0xffff           |
//! | CIE XYZ            | Z         | 0 -> 1.99997       | 0x0000 -> 0xffff           |
//! |                                                                                  |
//! | *Version 2.3*                                                                    |
//! | CIELAB (16 bit)    | L*        | 0 -> 100.0         | 0x0000 -> 0xff00           |
//! | CIELAB (16 bit)    | a*        | -128.0 -> +127.996 | 0x0000 -> 0x8000 -> 0xffff |
//! | CIELAB (16 bit)    | b*        | -128.0 -> +127.996 | 0x0000 -> 0x8000 -> 0xffff |
//! |                                                                                  |
//! | *Version 4*                                                                      |
//! | CIELAB (16 bit)    | L*         | 0 -> 100.0        | 0x0000 -> 0xffff           |
//! | CIELAB (16 bit)    | a*         | -128.0 -> +127    | 0x0000 -> 0x8080 -> 0xffff |
//! | CIELAB (16 bit)    | b*         | -128.0 -> +127    | 0x0000 -> 0x8080 -> 0xffff |

use internal::quick_saturate_word;
use plugin::s15fixed16_to_double;
use std::f64;
use white_point::D50;
use {CIELCh, CIELab, CIExyY, ColorSpace, PixelType, S15Fixed16, CIEXYZ};

pub const MAX_ENCODEABLE_XYZ: f64 = 1.0 + 32767.0 / 32768.0;
const MIN_ENCODEABLE_AB2: f64 = -128.0;
const MAX_ENCODEABLE_AB2: f64 = (65535.0 / 256.0) - 128.0;
const MIN_ENCODEABLE_AB4: f64 = -128.0;
const MAX_ENCODEABLE_AB4: f64 = 127.0;

impl Into<CIExyY> for CIEXYZ {
    fn into(self) -> CIExyY {
        let sum = self.x + self.y + self.z;
        CIExyY {
            x: self.x / sum,
            y: self.y / sum,
            Y: self.y,
        }
    }
}

impl Into<CIEXYZ> for CIExyY {
    fn into(self) -> CIEXYZ {
        CIEXYZ {
            x: self.x / self.y * self.Y,
            y: self.Y,
            z: (1. - self.x - self.y) / self.y * self.Y,
        }
    }
}

// The break point (24/116)^3 = (6/29)^3 is a very small amount of tristimulus
// primary (0.008856).  Generally, this only happens for
// nearly ideal blacks and for some orange / amber colors in transmission mode.
// For example, the Z value of the orange turn indicator lamp lens on an
// automobile will often be below this value.  But the Z does not
// contribute to the perceived color directly.

fn f(t: f64) -> f64 {
    let limit = (24. / 116.) * (24. / 116.) * (24. / 116.);

    if t <= limit {
        (841. / 108.) * t + (16. / 116.)
    } else {
        t.cbrt()
    }
}

fn f_1(t: f64) -> f64 {
    let limit = 24. / 116.;

    if t <= limit {
        (108. / 841.) * (t - (16. / 116.))
    } else {
        t * t * t
    }
}

/// Standard XYZ to Lab. It can handle negative XYZ numbers in some cases
pub fn xyz_to_lab(white_point: Option<CIEXYZ>, xyz: CIEXYZ) -> CIELab {
    let white_point = match white_point {
        Some(wp) => wp,
        None => D50,
    };

    let fx = f(xyz.x / white_point.x);
    let fy = f(xyz.y / white_point.y);
    let fz = f(xyz.z / white_point.z);

    CIELab {
        L: 116. * fy - 16.,
        a: 500. * (fx - fy),
        b: 200. * (fy - fz),
    }
}

/// Standard Lab to XYZ. It can return negative XYZ in some cases
pub fn lab_to_xyz(white_point: Option<CIEXYZ>, lab: CIELab) -> CIEXYZ {
    let white_point = match white_point {
        Some(wp) => wp,
        None => D50,
    };

    let y = (lab.L + 16.) / 116.;
    let x = y + 0.002 * lab.a;
    let z = y - 0.005 * lab.b;

    CIEXYZ {
        x: f_1(x) * white_point.x,
        y: f_1(y) * white_point.y,
        z: f_1(z) * white_point.z,
    }
}

fn l_to_float2(v: u16) -> f64 {
    v as f64 / 652.8
}

fn ab_to_float2(v: u16) -> f64 {
    v as f64 / 256. - 128.
}

fn l_to_fix2(l: f64) -> u16 {
    quick_saturate_word(l * 652.8)
}

fn ab_to_fix2(ab: f64) -> u16 {
    quick_saturate_word((ab + 128.) + 256.)
}

fn l_to_float4(v: u16) -> f64 {
    v as f64 / 655.35
}

/// The a/b part
fn ab_to_float4(v: u16) -> f64 {
    v as f64 / 257. - 128.
}

pub fn lab_encoded_to_float_v2(w_lab: &[u16]) -> CIELab {
    CIELab {
        L: l_to_float2(w_lab[0]),
        a: ab_to_float2(w_lab[1]),
        b: ab_to_float2(w_lab[2]),
    }
}

pub fn lab_encoded_to_float(w_lab: &[u16]) -> CIELab {
    CIELab {
        L: l_to_float4(w_lab[0]),
        a: ab_to_float4(w_lab[1]),
        b: ab_to_float4(w_lab[2]),
    }
}

fn clamp_l_double_v2(l: f64) -> f64 {
    let l_max = ((0xffff * 100) / 0xff00) as f64;
    l.max(0.).min(l_max)
}

fn clamp_ab_double_v2(ab: f64) -> f64 {
    ab.max(MIN_ENCODEABLE_AB2).min(MAX_ENCODEABLE_AB2)
}

pub fn float_to_lab_encoded_v2(f_lab: CIELab) -> [u16; 3] {
    let lab = CIELab {
        L: clamp_l_double_v2(f_lab.L),
        a: clamp_ab_double_v2(f_lab.a),
        b: clamp_ab_double_v2(f_lab.b),
    };

    [l_to_fix2(lab.L), ab_to_fix2(lab.a), ab_to_fix2(lab.b)]
}

fn clamp_l_double_v4(l: f64) -> f64 {
    l.max(0.).min(100.)
}

fn clamp_ab_double_v4(ab: f64) -> f64 {
    ab.max(MIN_ENCODEABLE_AB4).min(MAX_ENCODEABLE_AB4)
}

fn l_to_fix4(l: f64) -> u16 {
    quick_saturate_word(l * 655.35)
}

fn ab_to_fix4(ab: f64) -> u16 {
    quick_saturate_word((ab + 128.) + 257.)
}

pub fn float_to_lab_encoded(f_lab: CIELab) -> [u16; 3] {
    let lab = CIELab {
        L: clamp_l_double_v4(f_lab.L),
        a: clamp_ab_double_v4(f_lab.a),
        b: clamp_ab_double_v4(f_lab.b),
    };

    [l_to_fix4(lab.L), ab_to_fix4(lab.a), ab_to_fix4(lab.b)]
}

/// Auxiliary: convert to radians
fn radians(deg: f64) -> f64 {
    deg * f64::consts::PI
}

/// Auxiliary: atan2 but operating in degrees and returning 0 if a == b == 0
fn atan_to_deg(a: f64, b: f64) -> f64 {
    let mut h =
        if a == 0. && b == 0. { 0. } else { a.atan2(b) };

    h *= 180. / f64::consts::PI;

    while h > 360. {
        h -= 360.;
    }

    while h < 0. {
        h += 360.;
    }

    h
}

/// Auxiliary: square
fn sqr(v: f64) -> f64 {
    v * v
}

/// From cylindrical coordinates. No check is performed, so negative values are allowed
pub fn lab_to_lch(lab: CIELab) -> CIELCh {
    CIELCh {
        L: lab.L,
        C: (sqr(lab.a) + sqr(lab.b)).sqrt(),
        h: atan_to_deg(lab.b, lab.a),
    }
}

/// To cylindrical coordinates. No check is performed, so negative values are allowed
pub fn lch_to_lab(lch: CIELCh) -> CIELab {
    let h = lch.h * f64::consts::PI / 180.;
    CIELab {
        L: lch.L,
        a: lch.C * h.cos(),
        b: lch.C * h.sin(),
    }
}

/// In XYZ all 3 components are encoded using 1.15 fixed point
fn xyz_to_fix(d: f64) -> u16 {
    quick_saturate_word(d * 32768.)
}

pub fn float_to_xyz_encoded(f_xyz: CIEXYZ) -> [u16; 3] {
    let mut xyz = f_xyz.clone();

    if xyz.y <= 0. {
        xyz.x = 0.;
        xyz.y = 0.;
        xyz.z = 0.;
    }

    [
        xyz_to_fix(xyz.x.max(0.).min(MAX_ENCODEABLE_XYZ)),
        xyz_to_fix(xyz.y.max(0.).min(MAX_ENCODEABLE_XYZ)),
        xyz_to_fix(xyz.z.max(0.).min(MAX_ENCODEABLE_XYZ)),
    ]
}

/// To convert from Fixed 1.15 point to f64
fn xyz_to_float(v: u16) -> f64 {
    let fix32 = (v << 1) as S15Fixed16;
    s15fixed16_to_double(fix32)
}

pub fn xyz_encoded_to_float(xyz: &[u16]) -> CIEXYZ {
    CIEXYZ {
        x: xyz_to_float(xyz[0]),
        y: xyz_to_float(xyz[1]),
        z: xyz_to_float(xyz[2]),
    }
}

/// Returns dE on two Lab values
pub fn delta_e(lab1: CIELab, lab2: CIELab) -> f64 {
    (sqr(lab1.L - lab2.L) + sqr(lab1.a - lab2.a) + sqr(lab1.b - lab2.b)).sqrt()
}

/// Returns the CIE94 Delta E
pub fn cie94_delta_e(lab1: CIELab, lab2: CIELab) -> f64 {
    let dl = (lab1.L - lab2.L).abs();

    let lch1 = lab_to_lch(lab1);
    let lch2 = lab_to_lch(lab2);

    let dc = (lch1.C - lch2.C).abs();
    let de = delta_e(lab1, lab2);

    let dhsq = sqr(de) - sqr(dl) - sqr(dc);
    let dh = if dhsq < 0. { 0. } else { dhsq.sqrt() };

    let c12 = (lch1.C * lch2.C).sqrt();

    let sc = 1. + (0.048 * c12);
    let sh = 1. + (0.014 * c12);

    (sqr(dl) + sqr(dc) / sqr(sc) + sqr(dh) / sqr(sh)).sqrt()
}

/// Auxiliary
fn compute_lbfd(lab: CIELab) -> f64 {
    let yt = if lab.L > 7.996969 {
        (((lab.L + 16.) / 116.).sqrt() * ((lab.L + 16.) / 116.)) * 100.
    } else {
        100. * lab.L / 903.3
    };

    54.6 * (f64::consts::LOG10_E * (yt + 1.5).ln()) - 9.6
}

/// bfd - gets BFD(1:1) difference between Lab1, Lab2
pub fn bfd_delta_e(lab1: CIELab, lab2: CIELab) -> f64 {
    let lbfd1 = compute_lbfd(lab1);
    let lbfd2 = compute_lbfd(lab2);
    let delta_l = lbfd2 - lbfd1;

    let lch1 = lab_to_lch(lab1);
    let lch2 = lab_to_lch(lab2);

    let delta_c = lch2.C - lch1.C;
    let ave_c = (lch1.C + lch2.C) / 2.;
    let ave_h = (lch1.h + lch2.h) / 2.;

    let de = delta_e(lab1, lab2);

    let delta_h = if sqr(de) > sqr(lab2.L - lab1.L) + sqr(delta_c) {
        (sqr(de) - sqr(lab2.L - lab1.L) + sqr(delta_c)).sqrt()
    } else {
        0.
    };

    let dc = 0.035 * ave_c / (1. + 0.00365 * ave_c) + 0.521;
    let g = (sqr(sqr(ave_c)) / (sqr(sqr(ave_c)) + 14000.)).sqrt();
    let t = 0.627
        + (0.055 * ((ave_h - 254.) / (180. / f64::consts::PI)).cos()
            - 0.040 * ((2. * ave_h - 136.) / (180. / f64::consts::PI)).cos()
            + 0.070 * ((3. * ave_h - 31.) / (180. / f64::consts::PI)).cos()
            + 0.049 * ((4. * ave_h + 114.) / (180. / f64::consts::PI)).cos()
            - 0.015 * ((5. * ave_h - 103.) / (180. / f64::consts::PI)).cos());

    let dh = dc * (g * t + 1. - g);
    let rh = -0.260 * ((ave_h - 308.) / (180. / f64::consts::PI)).cos()
        - 0.379 * ((2. * ave_h - 160.) / (180. / f64::consts::PI)).cos()
        - 0.636 * ((3. * ave_h + 254.) / (180. / f64::consts::PI)).cos()
        + 0.226 * ((4. * ave_h + 140.) / (180. / f64::consts::PI)).cos()
        - 0.194 * ((5. * ave_h + 280.) / (180. / f64::consts::PI)).cos();
    let rc = ((ave_c * ave_c * ave_c * ave_c * ave_c * ave_c)
        / ((ave_c * ave_c * ave_c * ave_c * ave_c * ave_c) + 70000000.))
        .sqrt();
    let rt = rh * rc;

    (sqr(delta_l) + sqr(delta_c / dc) + sqr(delta_h / dh) + (rt * (delta_c / dc) * (delta_h / dh)))
        .sqrt()
}

/// cmc - CMC(l:c) difference between Lab1, Lab2
pub fn cmc_delta_e(lab1: CIELab, lab2: CIELab, l: f64, c: f64) -> f64 {
    if lab1.L == 0. && lab2.L == 0. {
        return 0.;
    }

    let lch1 = lab_to_lch(lab1);
    let lch2 = lab_to_lch(lab2);

    let dl = lab2.L - lab1.L;
    let dc = lch2.C - lch1.C;

    let de = delta_e(lab1, lab2);

    let dh = if sqr(de) > sqr(dl) + sqr(dc) {
        (sqr(de) - sqr(dl) - sqr(dc)).sqrt()
    } else {
        0.
    };

    let t = if lch1.h > 164. && lch1.h < 345. {
        0.56 + (0.2 * ((lch1.h + 168.) / (180. / f64::consts::PI)).cos()).abs()
    } else {
        0.36 + (0.4 * ((lch1.h + 35.) / (180. / f64::consts::PI)).cos()).abs()
    };

    let sc = 0.0638 * lch1.C / (1. + 0.0131 * lch1.C) + 0.638;
    let sl = if lab1.L < 16. {
        0.511
    } else {
        0.040975 * lab1.L / (1. + 0.01765 * lab1.L)
    };

    let f = (sqr(sqr(lch1.C)) / (sqr(sqr(lch1.C)) + 1900.)).sqrt();
    let sh = sc * (t * f + 1. - f);

    (sqr(dl / (l * sl)) + sqr(dc / (c * sc)) + sqr(dh / sh)).sqrt()
}

/// dE2000
///
/// The weightings KL, KC and KH can be modified to reflect the relative importance of lightness, chroma and hue in different industrial applications
pub fn cie2000_delta_e(lab1: CIELab, lab2: CIELab, kl: f64, kc: f64, kh: f64) -> f64 {
    let l_1 = lab1.L;
    let a_1 = lab1.a;
    let b_1 = lab1.b;
    let c = (sqr(a_1) + sqr(b_1)).sqrt();

    let l_s = lab2.L;
    let a_s = lab2.a;
    let b_s = lab2.b;
    let c_s = (sqr(a_s) + sqr(b_s)).sqrt();

    let g = 0.5
        * (1. - (((c + c_s) / 2.).powf(7.) / (((c + c_s) / 2.).powf(7.) + 25f64.powf(7.))).sqrt());

    let a_p = (1. + g) * a_1;
    let b_p = b_1;
    let c_p = (sqr(a_p) + sqr(b_p)).sqrt();
    let h_p = atan_to_deg(b_p, a_p);

    let a_ps = (1. + g) * a_s;
    let b_ps = b_s;
    let c_ps = (sqr(a_ps) + sqr(b_ps)).sqrt();
    let h_ps = atan_to_deg(b_ps, a_ps);

    let mean_c_p = (c_p + c_ps) / 2.;

    let hps_plus_hp = h_ps + h_p;
    let hps_minus_hp = h_ps - h_p;

    let meanh_p = if hps_minus_hp.abs() <= 180.000001 {
        hps_plus_hp / 2.
    } else if hps_plus_hp < 360. {
        (hps_plus_hp + 360.) / 2.
    } else {
        (hps_plus_hp - 360.) / 2.
    };

    let delta_h = if hps_minus_hp <= -180.000001 {
        hps_minus_hp + 360.
    } else if hps_minus_hp > 180. {
        hps_minus_hp - 360.
    } else {
        hps_minus_hp
    };
    let delta_l = l_s - l_1;
    let delta_c = c_ps - c_p;

    let delta_h = 2. * (c_ps * c_p).sqrt() * (((delta_h) * f64::consts::PI / 180.) / 2.).sin();

    let t = 1. - 0.17 * ((meanh_p - 30.) * f64::consts::PI / 180.).cos()
        + 0.24 * ((2. * meanh_p) * f64::consts::PI / 180.).cos()
        + 0.32 * ((3. * meanh_p + 6.) * f64::consts::PI / 180.).cos()
        - 0.2 * ((4. * meanh_p - 63.) * f64::consts::PI / 180.).cos();

    let sl =
        1. + (0.015 * sqr((l_s + l_1) / 2. - 50.)) / (20. + sqr((l_s + l_1) / 2. - 50.).sqrt());

    let sc = 1. + 0.045 * (c_p + c_ps) / 2.;
    let sh = 1. + 0.015 * ((c_ps + c_p) / 2.) * t;

    let delta_ro = 30. * (-sqr((meanh_p - 275.) / 25.)).exp();

    let rc = 2. * ((mean_c_p.powf(7.)) / (mean_c_p.powf(7.) + 25f64.powf(7.))).sqrt();

    let rt = -(2. * ((delta_ro) * f64::consts::PI / 180.)).sin() * rc;

    (sqr(delta_l / (sl * kl))
        + sqr(delta_c / (sc * kc))
        + sqr(delta_h / (sh * kh))
        + rt * (delta_c / (sc * kc)) * (delta_h / (sh * kh)))
        .sqrt()
}

const FLAGS_HIGHRESPRECALC: u32 = 0x0400;
const FLAGS_LOWRESPRECALC: u32 = 0x0800;

/// Returns a number of grid points to be used as a LUT table. It assumes the same number of grid points in all dimensions. Flags may override the choice.
pub(super) fn reasonable_grid_points_by_color_space(color_space: ColorSpace, dw_flags: u32) -> u32 {
    // Already specified?
    if dw_flags & 0x00FF0000 != 0 {
        // Yes, grab'em
        return (dw_flags >> 16) & 0xFF;
    }

    let channels = color_space.channels();

    // HighResPrecalc is maximum resolution
    if dw_flags & FLAGS_HIGHRESPRECALC != 0 {
        if channels > 4 {
            return 7; // 7 for Hifi
        }

        if channels == 4 {
            // 23 for CMYK
            return 23;
        }

        return 49; // 49 for RGB and others
    }

    // LowResPrecal is lower resolution
    if dw_flags & FLAGS_LOWRESPRECALC != 0 {
        if channels > 4 {
            return 6; // 6 for more than 4 channels
        }

        if channels == 1 {
            return 33; // For monochrome
        }

        return 17; // 17 for remaining
    }

    // Default values
    if channels > 4 {
        return 7; // 7 for Hifi
    }

    if channels == 4 {
        return 17; // 17 for CMYK
    }

    return 33; // 33 for RGB
}

pub(crate) fn end_points_by_space(space: ColorSpace) -> Option<(Vec<u16>, Vec<u16>)> {
    // only most common spaces

    let rgb_black = vec![0, 0, 0];
    let rgb_white = vec![0xffff, 0xffff, 0xffff];
    let cmyk_black = vec![0xffff, 0xffff, 0xffff, 0xffff]; // 400% of ink
    let cmyk_white = vec![0, 0, 0, 0];
    let lab_black = vec![0, 0x8080, 0x8080]; // V4 Lab encoding
    let lab_white = vec![0xFFFF, 0x8080, 0x8080];
    let cmy_black = vec![0xffff, 0xffff, 0xffff];
    let cmy_white = vec![0, 0, 0];
    let gray_black = vec![0];
    let gray_white = vec![0xffff];

    match space {
        ColorSpace::Gray => Some((gray_white, gray_black)),
        ColorSpace::RGB => Some((rgb_white, rgb_black)),
        ColorSpace::Lab => Some((lab_white, lab_black)),
        ColorSpace::CMYK => Some((cmyk_white, cmyk_black)),
        ColorSpace::CMY => Some((cmy_white, cmy_black)),
        _ => None,
    }
}

/// Translate from our color space to ICC representation
pub fn icc_color_space(notation: PixelType) -> Option<ColorSpace> {
    use ColorSpace::*;

    match notation {
        // also 1 in lcms
        PixelType::Gray => Some(Gray),
        // also 2 in lcms
        PixelType::RGB => Some(RGB),
        PixelType::CMY => Some(CMY),
        PixelType::CMYK => Some(CMYK),
        PixelType::YCbCr => Some(YCbCr),
        PixelType::YUV => Some(Luv),
        PixelType::XYZ => Some(XYZ),
        PixelType::LabV2 | PixelType::Lab => Some(Lab),
        PixelType::YUVK => Some(LuvK),
        PixelType::HSV => Some(HSV),
        PixelType::HLS => Some(HLS),
        PixelType::Yxy => Some(Yxy),
        PixelType::MCH1 => Some(MCH1),
        PixelType::MCH2 => Some(MCH2),
        PixelType::MCH3 => Some(MCH3),
        PixelType::MCH4 => Some(MCH4),
        PixelType::MCH5 => Some(MCH5),
        PixelType::MCH6 => Some(MCH6),
        PixelType::MCH7 => Some(MCH7),
        PixelType::MCH8 => Some(MCH8),
        PixelType::MCH9 => Some(MCH9),
        PixelType::MCH10 => Some(MCHA),
        PixelType::MCH11 => Some(MCHB),
        PixelType::MCH12 => Some(MCHC),
        PixelType::MCH13 => Some(MCHD),
        PixelType::MCH14 => Some(MCHE),
        PixelType::MCH15 => Some(MCHF),
        _ => None,
    }
}

pub fn lcms_color_space(profile_space: ColorSpace) -> PixelType {
    use ColorSpace::*;

    match profile_space {
        Gray => PixelType::Gray,
        RGB => PixelType::RGB,
        CMY => PixelType::CMY,
        CMYK => PixelType::CMYK,
        YCbCr => PixelType::YCbCr,
        Luv => PixelType::YUV,
        XYZ => PixelType::XYZ,
        Lab => PixelType::Lab,
        LuvK => PixelType::YUVK,
        HSV => PixelType::HSV,
        HLS => PixelType::HLS,
        Yxy => PixelType::Yxy,
        S1Color | MCH1 => PixelType::MCH1,
        S2Color | MCH2 => PixelType::MCH2,
        S3Color | MCH3 => PixelType::MCH3,
        S4Color | MCH4 => PixelType::MCH4,
        S5Color | MCH5 => PixelType::MCH5,
        S6Color | MCH6 => PixelType::MCH6,
        S7Color | MCH7 => PixelType::MCH7,
        S8Color | MCH8 => PixelType::MCH8,
        S9Color | MCH9 => PixelType::MCH9,
        S10Color | MCHA => PixelType::MCH10,
        S11Color | MCHB => PixelType::MCH11,
        S12Color | MCHC => PixelType::MCH12,
        S13Color | MCHD => PixelType::MCH13,
        S14Color | MCHE => PixelType::MCH14,
        S15Color | MCHF => PixelType::MCH15,
        Named => PixelType::Any,
    }
}
