//! Gamma tone curves.
//!
//! Tone curves are powerful constructs that can contain curves specified in diverse ways.
//! The curve is stored in segments, where each segment can be sampled or specified by parameters.
//! A 16-bit simplification of the *whole* curve is kept for optimization purposes.
//! For float operations, each segment is evaluated separately.
//!
//! <s>Plug-ins may be used to define new parametric schemes, each plug-in may define up to
//! MAX_TYPES_IN_LCMS_PLUGIN functions types. For defining a function, the plug-in should provide
//! the type id, how many parameters each type has, and a pointer to a procedure that evaluates the
//! function. In the case of reverse evaluation, the evaluator will be called with the type id as a
//! negative value, and a sampled version of the reversed curve will be built.</s>

use internal::{quick_saturate_word, MATRIX_DET_TOLERANCE};
use op::ScalarOp;
use std::cmp::Ordering;
use std::sync::Arc;
use std::{f32, f64, fmt};

const MAX_NODES_IN_CURVE: u32 = 4097;
const MINUS_INF: f64 = -f64::INFINITY;
const PLUS_INF: f64 = f64::INFINITY;

/// Evaluator callback for user-supplied parametric curves. May implement more than one type.
pub type ParametricCurveEvaluator = fn(i32, &[f64]) -> Vec<ScalarOp>;

/// List item of supported parametric curves
#[derive(Clone)]
struct ParametricCurveCollection {
    /// Identification types.
    function_types: Vec<i32>,

    /// Number of parameters for each function.
    parameters: Vec<u32>,

    evaluator: Arc<ParametricCurveEvaluator>,
}

fn default_curves() -> Vec<ParametricCurveCollection> {
    vec![ParametricCurveCollection {
        function_types: vec![1, 2, 3, 4, 5, 6, 7, 8, 108],
        parameters: vec![1, 3, 4, 5, 7, 4, 5, 5, 1],
        evaluator: Arc::new(default_eval_parametric_fn),
    }]
}

// - DupPluginCurvesList
// - AllocCurvesPluginChunk
// - RegisterParametricCurvesPlugin
// - ParametricCurvesCollection
// - IsInSet

fn parametric_curve_by_type(p_type: i32, index: &mut usize) -> Option<ParametricCurveCollection> {
    let p_type = p_type.abs();
    for collection in default_curves().into_iter() {
        for i in 0..collection.function_types.len() {
            if collection.function_types[i] == p_type {
                *index = i;
                return Some(collection);
            }
        }
    }

    None
}

/// A tone curve segment.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveSegment {
    /// Domain; defined as domain.0 < x <= domain.1
    pub domain: (f32, f32),
    /// Parametric type.
    pub p_type: i32,
    /// Parameters if type != 0.
    pub params: [f64; 10],
    /// Array of floats if type == 0.
    pub sampled_points: Vec<f32>,
}

/// A tone curve.
///
/// See module-level documentation for details.
#[derive(Clone)]
pub struct ToneCurve {
    /// The tone curve segments.
    segments: Vec<(CurveSegment, Option<Arc<ParametricCurveEvaluator>>)>,

    /// This 16-bit table contains a limited precision representation of the whole curve and is kept
    /// for increasing xput on certain operations.
    table16: Vec<u16>,
}

impl PartialEq for ToneCurve {
    fn eq(&self, other: &ToneCurve) -> bool {
        if self.segments.len() != other.segments.len() {
            return false;
        }
        if self.table16 != other.table16 {
            return false;
        }
        for (a, b) in self.segments.iter().zip(other.segments.iter()) {
            if a.0 != b.0 {
                return false;
            }
        }
        true
    }
}

impl fmt::Debug for ToneCurve {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ToneCurve {{ {} segments, {} table entries }}",
            self.segments.len(),
            self.table16.len()
        )
    }
}

impl ToneCurve {
    /// Creates an empty gamma curve, by using tables. This only specifies the limited-precision
    /// part, and leaves the floating point description empty.
    pub fn new_table(values: Vec<u16>) -> Result<ToneCurve, String> {
        ToneCurve::new(Vec::new(), values)
    }

    /// Creates a segmented gamma curve and fills the table.
    pub fn new_segmented(segments: Vec<CurveSegment>) -> Result<ToneCurve, String> {
        let grid_points = if segments.len() == 1 && segments[0].p_type == 1 {
            // optimization for identity curves
            entries_by_gamma(segments[0].params[0])
        } else {
            4096
        };

        let mut values = Vec::new();
        values.resize(grid_points, 0);

        let mut tone_curve = ToneCurve::new(segments, values)?;

        // Once we have the floating point version, we can approximate a 16 bit table of 4096 entries
        // for performance reasons. This table would normally not be used except on 8/16 bits transforms.
        for i in 0..grid_points {
            let r = i as f64 / (grid_points - 1) as f64;
            let val = tone_curve.eval_segmented(r);
            // round and saturate
            tone_curve.table16[i] = quick_saturate_word(val * 65535.);
        }

        Ok(tone_curve)
    }

    /// Uses a segmented curve to store the floating point table.
    pub fn new_tabulated(values: Vec<f32>) -> Result<ToneCurve, String> {
        // A segmented tone curve should have function segments in the first and last positions
        let last_value = *values.last().unwrap_or(&0.);
        ToneCurve::new_segmented(vec![
            CurveSegment {
                domain: (-f32::INFINITY, 0.),
                p_type: 6,
                params: [
                    1.,
                    0.,
                    0.,
                    *values.first().unwrap_or(&0.) as f64,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                ],
                sampled_points: Vec::new(),
            },
            CurveSegment {
                domain: (0., 1.),
                p_type: 0,
                params: [0.; 10],
                sampled_points: values,
            },
            CurveSegment {
                domain: (1., f32::INFINITY),
                p_type: 6,
                params: [1., 0., 0., last_value as f64, 0., 0., 0., 0., 0., 0.],
                sampled_points: Vec::new(),
            },
        ])
    }

    /// Creates a new parametric tone curve.
    ///
    /// - Parameters: Curve type, a, b, c, d, e, f
    /// - Curve type is the ICC type +1
    /// - If the type is negative, then the curve is analytically inverted
    pub fn new_parametric(p_type: i32, params: &[f64]) -> Result<ToneCurve, String> {
        let mut pos = 0;
        // TODO: make this an enum or something to eliminate error case
        let c = if let Some(c) = parametric_curve_by_type(p_type, &mut pos) {
            c
        } else {
            return Err("Invalid parametric curve type".into());
        };

        let param_count = c.parameters[pos] as usize;
        let mut curve_params = [0.; 10];

        for i in 0..param_count {
            curve_params[i] = params[i];
        }

        ToneCurve::new_segmented(vec![CurveSegment {
            domain: (-f32::INFINITY, f32::INFINITY),
            p_type,
            params: curve_params,
            sampled_points: Vec::new(),
        }])
    }

    /// Creates a gamma table based on a gamma constant.
    pub fn new_gamma(gamma: f64) -> Result<ToneCurve, String> {
        ToneCurve::new_parametric(1, &[gamma])
    }

    /// Creates a new ToneCurve with the given segments and table.
    ///
    /// Segments xor values can be empty, and there must be fewer than 65531 values.
    pub fn new(segments: Vec<CurveSegment>, values: Vec<u16>) -> Result<ToneCurve, String> {
        // We allow huge tables, which are then restricted for smoothing operations
        if values.len() > 65530 {
            return Err("Too many entries (> 65530)".into());
        }

        if segments.is_empty() && values.is_empty() {
            return Err("No segments or table entries".into());
        }

        // Allocate all required pointers, etc.
        let mut tone_curve = ToneCurve {
            segments: Vec::new(),
            table16: values,
        };

        // Initialize the segments stuff. The evaluator for each segment is located and a pointer to it
        // is placed in advance to maximize performance.
        for segment in segments {
            let p_type = segment.p_type;
            if p_type == 0 {
                // type 0 is a special marker for table-based curves
                // TODO: compute interp params
            }
            tone_curve.segments.push((
                segment,
                match parametric_curve_by_type(p_type, &mut 0) {
                    Some(collection) => Some(Arc::clone(&collection.evaluator)),
                    None => None,
                },
            ));
        }

        // TODO: interp params

        Ok(tone_curve)
    }

    /// Evaluate a segmented function for a single value. Returns -Infinity if no valid segment is
    /// found. If fn type is 0, performs interpolation on the table.
    fn eval_segmented(&self, x: f64) -> f64 {
        match self.segments.binary_search_by(|segment| {
            if segment.0.domain.0 >= x as f32 {
                Ordering::Greater
            } else if segment.0.domain.1 < x as f32 {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        }) {
            Ok(idx) => {
                let segment = &self.segments[idx];

                if segment.0.p_type == 0 {
                    // segment is sampled
                    let _x =
                        (x as f32 - segment.0.domain.0) / (segment.0.domain.1 - segment.0.domain.0);
                    // TODO: interpolation
                    unimplemented!()
                } else {
                    match segment.1 {
                        Some(ref eval) => {
                            ScalarOp::eval(&eval(segment.0.p_type, &segment.0.params), x)
                        }
                        None => MINUS_INF,
                    }
                }
            }
            Err(_) => MINUS_INF,
        }
    }

    /// Returns the table representation.
    pub fn estimated_table(&self) -> &[u16] {
        &self.table16
    }

    /// Evaluates the tone curve at the given input value.
    pub fn eval_16(&self, _v: u16) -> u16 {
        // TODO
        unimplemented!()
    }

    /// Evaluates the tone curve at the given input value using floats.
    pub fn eval_float(&self, v: f32) -> f32 {
        // Check for 16 bits table. If so, this is a limited-precision tone curve
        if self.segments.is_empty() {
            let input = quick_saturate_word((v * 65535.).into()) as u16;
            let out = self.eval_16(input);

            out as f32 / 65535.
        } else {
            self.eval_segmented(v.into()) as f32
        }
    }

    /// Reverses the curve with 4096 samples. (see `reverse_with_samples`)
    pub fn reverse(&self) -> Result<ToneCurve, String> {
        self.reverse_with_samples(4096)
    }

    /// Reverses the curve, either analytically (if possible) or creating a table with the given
    /// number of samples.
    pub fn reverse_with_samples(&self, _samples: u32) -> Result<ToneCurve, String> {
        // Try to reverse it analytically whatever possible
        if self.segments.len() == 1
            && self.segments[0].0.p_type > 0
            && parametric_curve_by_type(self.segments[0].0.p_type, &mut 0).is_some()
        {
            return Self::new_parametric(-self.segments[0].0.p_type, &self.segments[0].0.params);
        }

        // TODO
        // Nope, reverse the table.
        // let out = Self::new_tabulated(samples);

        // We want to know if this is an ascending or descending table
        // let ascending = !self.is_descending();

        /*
        // Iterate across Y axis
        if i in 0..samples {
            let y = (cmsFloat64Number) i * 65535.0 / (nResultSamples - 1);
            let y = i as f64 * 65535. / (samples - 1);

            let j = interval(y, self.table16, self.interp_params);

            // Find interval in which y is within.
            j = GetInterval(y, InCurve->Table16, InCurve->InterpParams);
            if (j >= 0) {


                // Get limits of interval
                x1 = InCurve ->Table16[j];
                x2 = InCurve ->Table16[j+1];

                y1 = (cmsFloat64Number) (j * 65535.0) / (InCurve ->nEntries - 1);
                y2 = (cmsFloat64Number) ((j+1) * 65535.0 ) / (InCurve ->nEntries - 1);

                // If collapsed, then use any
                if (x1 == x2) {

                    out ->Table16[i] = _cmsQuickSaturateWord(Ascending ? y2 : y1);
                    continue;

                } else {

                    // Interpolate
                    a = (y2 - y1) / (x2 - x1);
                    b = y2 - a * x2;
                }
            }

            out ->Table16[i] = _cmsQuickSaturateWord(a* y + b);
        } */

        unimplemented!()
    }
}

/// Parametric function using floating point
fn default_eval_parametric_fn(p_type: i32, params: &[f64]) -> Vec<ScalarOp> {
    use self::ScalarOp::*;

    match p_type {
        // y = x ^ g
        1 => vec![IfLtElse(
            0.,
            vec![if (params[0] - 1.).abs() < MATRIX_DET_TOLERANCE {
                Noop
            } else {
                Const(0.)
            }],
            vec![Pow(params[0])],
        )],
        // type 1 reversed: x = y ^ (1 / g)
        -1 => vec![IfLtElse(
            0.,
            vec![if (params[0] - 1.).abs() < MATRIX_DET_TOLERANCE {
                Noop
            } else {
                Const(0.)
            }],
            vec![if params[0].abs() < MATRIX_DET_TOLERANCE {
                Const(PLUS_INF)
            } else {
                Pow(1. / params[0])
            }],
        )],
        // CIE 122-1966
        // y = (ax + b) ^ g | x >= -b / a
        // y = 0            | else
        2 => vec![if params[1].abs() < MATRIX_DET_TOLERANCE {
            Const(0.)
        } else {
            IfLtElse(
                -params[2] / params[1],
                vec![Const(0.)],
                vec![
                    Mul(params[1]),
                    Add(params[2]),
                    IfLtElse(0., vec![Const(0.)], vec![Pow(params[0])]),
                ],
            )
        }],
        // type 2 reversed: x = (y ^ (1 / g) - b) / a
        -2 => vec![if params[0].abs() < MATRIX_DET_TOLERANCE
            || params[1].abs() < MATRIX_DET_TOLERANCE
        {
            Const(0.)
        } else {
            IfLtElse(
                0.,
                vec![Const(0.)],
                vec![
                    Pow(1. / params[0]),
                    Sub(params[2]),
                    Div(params[1]),
                    IfLtElse(0., vec![Const(0.)], vec![Noop]),
                ],
            )
        }],
        // IEC 61966-3
        // y = (ax + b) ^ g | x <= -b / a
        // y = c            | else
        3 => vec![if params[1].abs() < MATRIX_DET_TOLERANCE {
            Const(0.)
        } else {
            IfLtElse(
                (-params[2] / params[1]).max(0.),
                vec![Const(params[3])],
                vec![
                    Mul(params[1]),
                    Add(params[2]),
                    IfLtElse(0., vec![Const(0.)], vec![Pow(params[0]), Add(params[3])]),
                ],
            )
        }],
        // type 3 reversed
        // x = ((y - c) ^ (1 / g) - b) / a | y >= c
        // x = -b / a                      | else
        -3 => vec![if params[1].abs() < MATRIX_DET_TOLERANCE {
            Const(0.)
        } else {
            IfLtElse(
                params[3],
                vec![Const(-params[2] / params[1])],
                vec![
                    Sub(params[3]),
                    IfLtElse(
                        0.,
                        vec![Const(0.)],
                        vec![Pow(1. / params[0]), Sub(params[2]), Div(params[1])],
                    ),
                ],
            )
        }],
        // IEC 61966-2.1 (sRGB)
        // y = (ax + b) ^ g | x >= d
        // y = cx           | else
        4 => vec![IfLtElse(
            params[4],
            vec![Mul(params[3])],
            vec![
                Mul(params[1]),
                Add(params[2]),
                IfLtElse(0., vec![Const(0.)], vec![Pow(params[0])]),
            ],
        )],
        // type 4 reversed
        // x = (y ^ (1 / g) - b) / a | y >= (ad + b) ^ g
        // x = y / c                 | else
        -4 => vec![if params[0].abs() < MATRIX_DET_TOLERANCE
            || params[1].abs() < MATRIX_DET_TOLERANCE
            || params[3].abs() < MATRIX_DET_TOLERANCE
        {
            Const(0.)
        } else {
            IfLtElse(
                (params[1] * params[4] + params[2]).max(0.).powf(params[0]),
                vec![Div(params[3])],
                vec![Pow(1. / params[0]), Sub(params[2]), Div(params[1])],
            )
        }],
        // y = (ax + b) ^ g + e | x >= d
        // y = cx + f           | else
        5 => vec![IfLtElse(
            params[4],
            vec![Mul(params[3]), Add(params[6])],
            vec![
                Mul(params[1]),
                Add(params[2]),
                IfLtElse(
                    0.,
                    vec![Const(params[5])],
                    vec![Pow(params[0]), Add(params[5])],
                ),
            ],
        )],
        // reversed type 5
        // x = ((y - e) ^ (1 / g) - b) / a | y >= (ad + b) ^ g + e, cd + f
        // x = (y - f) / c                 | else
        -5 => vec![if params[0].abs() < MATRIX_DET_TOLERANCE
            || params[3].abs() < MATRIX_DET_TOLERANCE
        {
            Const(0.)
        } else {
            IfLtElse(
                params[3] * params[4] + params[6],
                vec![Sub(params[6]), Div(params[3])],
                vec![
                    Sub(params[5]),
                    IfLtElse(
                        0.,
                        vec![Const(0.)],
                        vec![Pow(1. / params[0]), Sub(params[2]), Div(params[1])],
                    ),
                ],
            )
        }],

        // Types 6,7,8 comes from segmented curves as described in ICCSpecRevision_02_11_06_Float.pdf

        // Type 6 is basically identical to type 5 without d
        // y = (ax + b) ^ g + c
        6 => vec![
            Mul(params[1]),
            Add(params[2]),
            IfLtElse(
                0.,
                vec![Const(params[3])],
                vec![Pow(params[0]), Add(params[3])],
            ),
        ],
        // type 6 reversed: ((y - c) ^ (1 / g) - b) / a
        -6 => if params[1].abs() < MATRIX_DET_TOLERANCE {
            vec![Const(0.)]
        } else {
            vec![
                Sub(params[3]),
                IfLtElse(
                    0.,
                    vec![Const(0.)],
                    vec![Pow(1. / params[0]), Sub(params[2]), Div(params[1])],
                ),
            ]
        },
        // y = a log_10(b * x ^ g + c) + d
        7 => vec![
            Pow(params[0]),
            Mul(params[2]),
            Add(params[3]),
            IfLeqElse(
                0.,
                vec![Const(params[4])],
                vec![Log10, Mul(params[1]), Add(params[4])],
            ),
        ],
        //                                (y - d) / a = log_10(b * x ^ g + c)
        //                       pow(10, (y - d) / a) = b * x ^ g + c
        // pow((pow(10, (y - d) / a) - c) / b, 1 / g) = x
        -7 => if params[0].abs() < MATRIX_DET_TOLERANCE
            || params[1].abs() < MATRIX_DET_TOLERANCE
            || params[2].abs() < MATRIX_DET_TOLERANCE
        {
            vec![Const(0.)]
        } else {
            vec![
                Sub(params[4]),
                Div(params[1]),
                Exp(10.),
                Sub(params[3]),
                Div(params[2]),
                Pow(1. / params[0]),
            ]
        },
        // y = a * b ^ (cx + d) + e
        8 => vec![
            Mul(params[2]),
            Add(params[3]),
            Exp(params[1]),
            Mul(params[0]),
            Add(params[4]),
        ],
        // y = (log((y - e) / a) / log(b) - d) / c
        // params: [a, b, c, d, e]
        -8 => vec![
            Sub(params[4]),
            IfLtElse(
                0.,
                vec![Const(0.)],
                if params[0].abs() < MATRIX_DET_TOLERANCE || params[2].abs() < MATRIX_DET_TOLERANCE
                {
                    vec![Const(0.)]
                } else {
                    vec![
                        Div(params[0]),
                        Log10,
                        Div(params[1].log10()),
                        Sub(params[3]),
                        Div(params[2]),
                    ]
                },
            ),
        ],
        // S-Shaped: (1 - (1 - x) ^ (1 / g)) ^ (1 / g)
        108 => if params[0].abs() < MATRIX_DET_TOLERANCE {
            vec![Const(0.)]
        } else {
            vec![
                Mul(-1.),
                Add(1.),
                Pow(1. / params[0]),
                Mul(-1.),
                Add(1.),
                Pow(1. / params[0]),
            ]
        },
        //                   y = (1 - (1 - x) ^ (1 / g)) ^ (1 / g)
        //               y ^ g = (1 - (1 - x) ^ (1 / g))
        //           1 - y ^ g = (1 - x) ^ (1 / g)
        //     (1 - y ^ g) ^ g = 1 - x
        // 1 - (1 - y ^ g) ^ g = x
        -108 => vec![
            Pow(params[0]),
            Mul(-1.),
            Add(1.),
            Pow(params[0]),
            Mul(-1.),
            Add(1.),
        ],
        // unsupported
        _ => vec![Const(0.)],
    }
}

fn entries_by_gamma(gamma: f64) -> usize {
    if (gamma - 1.) < 0.001 {
        2
    } else {
        4096
    }
}

// TODO: JoinToneCurve
// TODO: GetInterval
// TODO: ReverseToneCurveEx
// TODO: ReverseToneCurve
// TODO: smooth2
// TODO: SmoothToneCurve
// TODO: IsToneCurveLinear
// TODO: IsToneCurveMonotonic
// TODO: IsToneCurveDescending
// TODO: IsToneCurveMultisegment
// TODO: GetToneCurveParametricType
// TODO: EvalToneCurveFloat
// TODO: EvalToneCurve16
// TODO: EstimateGamma
