use super::*;
use crate::gamma::ToneCurve;
use crate::pixel_format::RGB;
use crate::profile::Profile;
use crate::transform::Transform;

macro_rules! assert_approx_eq_slices {
    ($lhs:expr, $rhs:expr, $epsilon:expr) => {{
        let is_eq = if $lhs.len() == $rhs.len() {
            let mut is_eq = true;
            for (a, b) in $lhs.iter().zip($rhs.iter()) {
                if (a - b).abs() > $epsilon {
                    is_eq = false;
                    break;
                }
            }
            is_eq
        } else {
            false
        };

        if !is_eq {
            panic!(
                "approx assertion failed: left == right\nleft: {:?}\nright:{:?}",
                $lhs, $rhs
            );
        }
    }};
}

#[test]
fn aces_cg_srgb_round_trip() {
    let aces_cg = Profile::new_rgb(
        CIExyY {
            x: 0.32168,
            y: 0.33767,
            Y: 1.,
        },
        CIExyYTriple {
            red: CIExyY {
                x: 0.713,
                y: 0.293,
                Y: 1.,
            },
            green: CIExyY {
                x: 0.165,
                y: 0.830,
                Y: 1.,
            },
            blue: CIExyY {
                x: 0.128,
                y: 0.044,
                Y: 1.,
            },
        },
        [
            ToneCurve::new_gamma(1.).unwrap(),
            ToneCurve::new_gamma(1.).unwrap(),
            ToneCurve::new_gamma(1.).unwrap(),
        ],
    )
    .unwrap();

    let srgb = Profile::new_srgb();

    let aces_to_srgb: Transform<RGB<f32>, RGB<f32>> =
        Transform::new(&aces_cg, &srgb, Intent::Perceptual).unwrap();
    let srgb_to_aces: Transform<RGB<f32>, RGB<f32>> =
        Transform::new(&srgb, &aces_cg, Intent::Perceptual).unwrap();

    let aces_px: [f32; 3] = [0.5, 0.1, 0.7];
    let mut srgb_px: [f32; 3] = [-1.; 3];
    let mut aces_px_2: [f32; 3] = [-1.; 3];

    aces_to_srgb.convert(&aces_px, &mut srgb_px);
    srgb_to_aces.convert(&srgb_px, &mut aces_px_2);

    println!(
        "ACEScg {:?} -> sRGB {:?} -> ACEScg {:?}",
        aces_px, srgb_px, aces_px_2
    );

    assert_approx_eq_slices!(srgb_px, [0.87143475, 0.22537376, 0.89735174], 0.000001);
    assert_approx_eq_slices!(aces_px, aces_px_2, 0.000001);
}

#[test]
fn deser_testbed() {
    use std::io::Cursor;

    const TEST1: &[u8] = include_bytes!("../lcms-testbed/test1.icc");
    const TEST2: &[u8] = include_bytes!("../lcms-testbed/test2.icc");
    const TEST3: &[u8] = include_bytes!("../lcms-testbed/test3.icc");
    const TEST4: &[u8] = include_bytes!("../lcms-testbed/test4.icc");
    const TEST5: &[u8] = include_bytes!("../lcms-testbed/test5.icc");
    const BAD: &[u8] = include_bytes!("../lcms-testbed/bad.icc");
    const TOO_SMALL: &[u8] = include_bytes!("../lcms-testbed/toosmall.icc");

    let test1 = Profile::deser(&mut Cursor::new(TEST1)).unwrap();
    println!("Test1: {:?}", test1);
    let test2 = Profile::deser(&mut Cursor::new(TEST2)).unwrap();
    println!("Test2: {:?}", test2);
    let test3 = Profile::deser(&mut Cursor::new(TEST3)).unwrap();
    println!("Test3: {:?}", test3);
    let test4 = Profile::deser(&mut Cursor::new(TEST4)).unwrap();
    println!("Test4: {:?}", test4);
    let test5 = Profile::deser(&mut Cursor::new(TEST5)).unwrap();
    println!("Test5: {:?}", test5);

    assert!(Profile::deser(&mut Cursor::new(BAD)).is_err());
    assert!(Profile::deser(&mut Cursor::new(TOO_SMALL)).is_err());
    assert!(Profile::deser(&mut Cursor::new(&[])).is_err());
}
