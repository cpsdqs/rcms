use pcs::*;
use super::*;
use gamma::ToneCurve;
use profile::Profile;
use transform::Transform;

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
            panic!("approx assertion failed: left == right\nleft: {:?}\nright:{:?}", $lhs, $rhs);
        }
    }}
}

#[test]
fn aces_cg_srgb_round_trip() {
    let aces_cg = Profile::new_rgb(
        xyz_to_xy_y(CIEXYZ {
            x: 0.95265,
            y: 1.00000,
            z: 1.00883,
        }),
        /*CIExyY {
            x: 0.32168,
            y: 0.33767,
            Y: 1.,
        },*/
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
    ).unwrap();

    let srgb = Profile::new_srgb().unwrap();
    println!("{:?}\n{:?}", aces_cg, srgb);

    let transform = Transform::new(&aces_cg, aces_cg.formatter_for_cs(4, true), &srgb, srgb.formatter_for_cs(4, true), Intent::Perceptual).unwrap();
    let transform_rev = Transform::new(&srgb, srgb.formatter_for_cs(4, true), &aces_cg, aces_cg.formatter_for_cs(4, true), Intent::Perceptual).unwrap();

    println!("\n{:?}\n", transform);

    let a: [f32; 3] = [0.5, 0.1, 0.7];
    let mut b: [f32; 3] = [-1.; 3];
    let mut c: [f32; 3] = [-1.; 3];

    unsafe { transform.convert_any(&a as *const _ as *const (), &mut b as *mut _ as *mut (), 1) };
    unsafe { transform_rev.convert_any(&b as *const _ as *const (), &mut c as *mut _ as *mut (), 1) };

    println!("ACEScg {:?} -> sRGB {:?} -> ACEScg {:?}", a, b, c);

    assert_approx_eq_slices!(&b, &[0.87143475, 0.22537376, 0.89735174], 0.000001);
    assert_approx_eq_slices!(a, c, 0.000001);
}
