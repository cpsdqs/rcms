//! White points.

use cgmath::{Matrix, Matrix3, SquareMatrix, Vector3};
use {CIExyY, CIExyYTriple, CIEXYZ};

/// D50 — widely used
pub const D50: CIEXYZ = CIEXYZ {
    x: 0.9642,
    y: 1.,
    z: 0.8249,
};

/// Correlates a black body chromaticity from the given temperature in Kelvin.
/// Valid input range is 4000–25000 K.
pub fn white_point_from_temp(temp_k: f64) -> Option<CIExyY> {
    let t = temp_k;
    let t2 = t * t;
    let t3 = t2 * t;

    let x = if t >= 4000. && t <= 7000. {
        // for correlated color temperature (T) between 4000K and 7000K:
        -4.6070 * (1e9 / t3) + 2.9678 * (1e6 / t2) + 0.09911 * (1e3 / t) + 0.244063
    } else if t > 7000. && t <= 25000. {
        // or for correlated color temperature (T) between 7000K and 25000K:
        -2.0064 * (1e9 / t3) + 1.9018 * (1e6 / t2) + 0.24748 * (1e3 / t) + 0.237040
    } else {
        return None;
    };

    let y = -3. * (x * x) + 2.87 * x + 0.275;

    // wave factors (not used, but here for futures extensions)

    // M1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / (0.0241 + 0.2562 * x - 0.7341 *
    // y); M2 = (0.03 - 31.4424 * x + 30.0717 * y) / (0.0241 + 0.2562 * x -
    // 0.7341 * y);

    Some(CIExyY { x, y, Y: 1. })
}

struct ISOTemperature {
    /// Temperature in microreciprocal kelvin.
    mirek: f64,
    /// U coordinate of intersection with blackbody locus.
    ut: f64,
    /// V coordinate of intersection with blackbody locus.
    vt: f64,
    /// Slope of temperature line.
    tt: f64,
}

const ISO_TEMP_DATA: [ISOTemperature; 31] = [
    ISOTemperature {
        mirek: 0.,
        ut: 0.18006,
        vt: 0.26352,
        tt: -0.24341,
    },
    ISOTemperature {
        mirek: 10.,
        ut: 0.18066,
        vt: 0.26589,
        tt: -0.25479,
    },
    ISOTemperature {
        mirek: 20.,
        ut: 0.18133,
        vt: 0.26846,
        tt: -0.26876,
    },
    ISOTemperature {
        mirek: 30.,
        ut: 0.18208,
        vt: 0.27119,
        tt: -0.28539,
    },
    ISOTemperature {
        mirek: 40.,
        ut: 0.18293,
        vt: 0.27407,
        tt: -0.30470,
    },
    ISOTemperature {
        mirek: 50.,
        ut: 0.18388,
        vt: 0.27709,
        tt: -0.32675,
    },
    ISOTemperature {
        mirek: 60.,
        ut: 0.18494,
        vt: 0.28021,
        tt: -0.35156,
    },
    ISOTemperature {
        mirek: 70.,
        ut: 0.18611,
        vt: 0.28342,
        tt: -0.37915,
    },
    ISOTemperature {
        mirek: 80.,
        ut: 0.18740,
        vt: 0.28668,
        tt: -0.40955,
    },
    ISOTemperature {
        mirek: 90.,
        ut: 0.18880,
        vt: 0.28997,
        tt: -0.44278,
    },
    ISOTemperature {
        mirek: 100.,
        ut: 0.19032,
        vt: 0.29326,
        tt: -0.47888,
    },
    ISOTemperature {
        mirek: 125.,
        ut: 0.19462,
        vt: 0.30141,
        tt: -0.58204,
    },
    ISOTemperature {
        mirek: 150.,
        ut: 0.19962,
        vt: 0.30921,
        tt: -0.70471,
    },
    ISOTemperature {
        mirek: 175.,
        ut: 0.20525,
        vt: 0.31647,
        tt: -0.84901,
    },
    ISOTemperature {
        mirek: 200.,
        ut: 0.21142,
        vt: 0.32312,
        tt: -1.0182,
    },
    ISOTemperature {
        mirek: 225.,
        ut: 0.21807,
        vt: 0.32909,
        tt: -1.2168,
    },
    ISOTemperature {
        mirek: 250.,
        ut: 0.22511,
        vt: 0.33439,
        tt: -1.4512,
    },
    ISOTemperature {
        mirek: 275.,
        ut: 0.23247,
        vt: 0.33904,
        tt: -1.7298,
    },
    ISOTemperature {
        mirek: 300.,
        ut: 0.24010,
        vt: 0.34308,
        tt: -2.0637,
    },
    ISOTemperature {
        mirek: 325.,
        ut: 0.24702,
        vt: 0.34655,
        tt: -2.4681,
    },
    ISOTemperature {
        mirek: 350.,
        ut: 0.25591,
        vt: 0.34951,
        tt: -2.9641,
    },
    ISOTemperature {
        mirek: 375.,
        ut: 0.26400,
        vt: 0.35200,
        tt: -3.5814,
    },
    ISOTemperature {
        mirek: 400.,
        ut: 0.27218,
        vt: 0.35407,
        tt: -4.3633,
    },
    ISOTemperature {
        mirek: 425.,
        ut: 0.28039,
        vt: 0.35577,
        tt: -5.3762,
    },
    ISOTemperature {
        mirek: 450.,
        ut: 0.28863,
        vt: 0.35714,
        tt: -6.7262,
    },
    ISOTemperature {
        mirek: 475.,
        ut: 0.29685,
        vt: 0.35823,
        tt: -8.5955,
    },
    ISOTemperature {
        mirek: 500.,
        ut: 0.30505,
        vt: 0.35907,
        tt: -11.324,
    },
    ISOTemperature {
        mirek: 525.,
        ut: 0.31320,
        vt: 0.35968,
        tt: -15.628,
    },
    ISOTemperature {
        mirek: 550.,
        ut: 0.32129,
        vt: 0.36011,
        tt: -23.325,
    },
    ISOTemperature {
        mirek: 575.,
        ut: 0.32931,
        vt: 0.36038,
        tt: -40.770,
    },
    ISOTemperature {
        mirek: 600.,
        ut: 0.33724,
        vt: 0.36051,
        tt: -116.45,
    },
];

/// Uses Robertson's method to attempt to obtain a temperature from a given white point.
pub fn temp_from_white_point(white_point: CIExyY) -> Option<f64> {
    let mut di = 0.;
    let mut mi = 0.;
    let xs = white_point.x;
    let ys = white_point.y;

    // convert (x, y) to CIE 1960 (u, white_point)
    let us = (2. * xs) / (-xs + 6. * ys + 1.5);
    let vs = (3. * ys) / (-xs + 6. * ys + 1.5);

    for j in 0..ISO_TEMP_DATA.len() {
        let uj = ISO_TEMP_DATA[j].ut;
        let vj = ISO_TEMP_DATA[j].vt;
        let tj = ISO_TEMP_DATA[j].tt;
        let mj = ISO_TEMP_DATA[j].mirek;

        let dj = ((vs - vj) - tj * (us - uj)) / (1. + tj * tj).sqrt();

        if j != 0 && di / dj < 0. {
            // found a match
            return Some(1_000_000. / (mi + (di / (di - dj)) * (mj - mi)));
        }

        di = dj;
        mi = mj;
    }

    // not found
    None
}

pub(crate) fn mat3_per(a: Matrix3<f64>, b: Matrix3<f64>) -> Matrix3<f64> {
    // LCMS Mat3per’s arguments are swapped
    b * a
}

pub(crate) fn mat3_eval(a: Matrix3<f64>, v: Vector3<f64>) -> Vector3<f64> {
    // LCMS matrix multiplication is transposed for some reason, so here’s an extra function
    // (this took me four hours to track down)
    a.transpose() * v
}

/// Computes a chromatic adaptation matrix using chad as the cone matrix
fn compute_chromatic_adaptation(
    chad: Matrix3<f64>,
    source_wp: CIEXYZ,
    dest_wp: CIEXYZ,
) -> Option<Matrix3<f64>> {
    let inverse = match chad.invert() {
        Some(inverse) => inverse,
        None => return None,
    };

    let cone_source_xyz: Vector3<f64> = source_wp.into();
    let cone_dest_xyz: Vector3<f64> = dest_wp.into();

    let cone_source_rgb = mat3_eval(chad, cone_source_xyz);
    let cone_dest_rgb = mat3_eval(chad, cone_dest_xyz);

    // build matrix
    let cone = Matrix3::from_diagonal(Vector3::new(
        cone_dest_rgb.x / cone_source_rgb.x,
        cone_dest_rgb.y / cone_source_rgb.y,
        cone_dest_rgb.z / cone_source_rgb.z,
    ));

    // normalize
    Some(mat3_per(inverse, mat3_per(cone, chad)))
}

/// Bradford matrix
const BRADFORD: Matrix3<f64> = Matrix3 {
    x: Vector3 {
        x: 0.8951,
        y: 0.2664,
        z: -0.1614,
    },
    y: Vector3 {
        x: -0.7502,
        y: 1.7135,
        z: 0.0367,
    },
    z: Vector3 {
        x: 0.0389,
        y: -0.0685,
        z: 1.0296,
    },
};

/// Returns the final chromatic adaptation from illuminant FromIll to Illuminant ToIll
///
/// The cone matrix can be specified in ConeMatrix. If None, Bradford is assumed
pub(super) fn adaptation_matrix(
    cone_matrix: Option<Matrix3<f64>>,
    from_ill: CIEXYZ,
    to_ill: CIEXYZ,
) -> Option<Matrix3<f64>> {
    let cone_matrix = cone_matrix.unwrap_or(BRADFORD);
    compute_chromatic_adaptation(cone_matrix, from_ill, to_ill)
}

/// Same as anterior, but assuming D50 destination. White point is given in xyY
fn adapt_matrix_to_d50(mat: Matrix3<f64>, source_wp: CIExyY) -> Option<Matrix3<f64>> {
    adaptation_matrix(None, source_wp.into(), D50).map(|bradford| mat3_per(bradford, mat))
}

/// Build a white point, primary chromas transfer matrix from RGB to CIE XYZ.
///
/// This is just an approximation, I am not handling all the non-linear
/// aspects of the RGB to XYZ process, and assumming that the gamma correction
/// has transitive property in the transformation chain.
///
/// the algorithm:
/// - First I build the absolute conversion matrix using primaries in XYZ. This
///   matrix is next inverted
/// - Then I eval the source white point across this matrix obtaining the
///   coefficients of the transformation
/// - Then, I apply these coefficients to the original matrix
pub(super) fn build_rgb_to_xyz_transfer_matrix(
    white_pt: CIExyY,
    primrs: CIExyYTriple,
) -> Option<Matrix3<f64>> {
    let xn = white_pt.x;
    let yn = white_pt.y;
    let xr = primrs.red.x;
    let yr = primrs.red.y;
    let xg = primrs.green.x;
    let yg = primrs.green.y;
    let xb = primrs.blue.x;
    let yb = primrs.blue.y;

    let primaries = Matrix3::from_cols(
        (xr, xg, xb).into(),
        (yr, yg, yb).into(),
        (1. - xr - yr, 1. - xg - yg, 1. - xb - yb).into(),
    );

    let result = match primaries.invert() {
        Some(result) => result,
        None => return None,
    };

    let white_point = Vector3::new(xn / yn, 1., (1. - xn - yn) / yn);

    // let coef = match solve_matrix(result, white_point) {
    //     Some(coef) => coef,
    //     None => return None,
    // };
    let coef = mat3_eval(result, white_point);

    adapt_matrix_to_d50(
        Matrix3::from_cols(
            (coef.x * xr, coef.y * xg, coef.z * xb).into(),
            (coef.x * yr, coef.y * yg, coef.z * yb).into(),
            (
                coef.x * (1. - xr - yr),
                coef.y * (1. - xg - yg),
                coef.z * (1. - xb - yb),
            ).into(),
        ),
        white_pt,
    )
}

/// Adapts a color to a given illuminant. Original color is expected to have
/// a source_wp white point.
pub fn adapt_to_illuminant(source_wp: CIEXYZ, illuminant: CIEXYZ, value: CIEXYZ) -> Option<CIEXYZ> {
    match adaptation_matrix(None, source_wp, illuminant) {
        Some(bradford) => {
            let vec = mat3_eval(bradford, Vector3::new(value.x, value.y, value.z));
            Some(CIEXYZ {
                x: vec.x,
                y: vec.y,
                z: vec.z,
            })
        }
        None => return None,
    }
}

pub(crate) fn solve_matrix(matrix: Matrix3<f64>, b: Vector3<f64>) -> Option<Vector3<f64>> {
    match matrix.invert() {
        Some(inverse) => Some(mat3_eval(inverse, b)),
        None => None,
    }
}
