//! CPU/GPU Operations.
//!
//! (not in LCMS)

use std::mem;

// TODO: see if this is necessary and if it can’t be replaced by a static shader and uniform buffers or something

/// Transform operations on scalar values.
///
/// Current value `x`, other value `y`.
///
/// This is essentially a tiny programming language.
/// Memory consists of two values `x` and `y`. All operations act upon `x`.
/// `y` is scope-specific and is reset to an *undefined value* in a new scope
/// (e.g. in an if-clause).
///
/// # Examples
/// ```
/// # use lcms_prime::ScalarOp;
/// let ops = vec![ScalarOp::Mul(2.), ScalarOp::Add(4.), ScalarOp::Pow(2.)];
///
/// assert_eq!(ScalarOp::eval(&ops, 12.), (12f64 * 2. + 4.).powf(2.));
/// let gpu_shader = ScalarOp::compile_shader(&ops, "x", "double");
/// assert_eq!(
///     gpu_shader,
///     "x *= 2.;
/// x += 4.;
/// x = pow(x, 2.);
/// "
/// );
/// ```
#[derive(Clone, PartialEq)]
pub enum ScalarOp {
    /// Does nothing.
    Noop,
    /// `x = v`.
    Const(f64),
    /// Assigns `y = x`.
    Dup,
    /// Swaps `x` and `y`.
    Swap,
    /// `if x < v` first set, else second set.
    IfLtElse(f64, Vec<ScalarOp>, Vec<ScalarOp>),
    /// `if x <= v` first set, else second set.
    IfLeqElse(f64, Vec<ScalarOp>, Vec<ScalarOp>),
    /// `x = x ^ v`.
    Pow(f64),
    /// `x = v ^ x`.
    Exp(f64),
    /// `x = x + v`.
    Add(f64),
    /// `x = x - v`.
    Sub(f64),
    /// `x = x * v`.
    Mul(f64),
    /// `x = x / v`.
    Div(f64),
    /// `x = log_10 x`.
    Log10,
}

fn format_decimal(n: f64) -> String {
    let mut formatted = format!("{}", n);
    if !formatted.contains(".") {
        formatted += ".";
    }
    formatted
}

fn indent(n: u8) -> String {
    "    ".repeat(n as usize)
}

impl ScalarOp {
    fn requires_y(&self) -> bool {
        match *self {
            ScalarOp::Noop => false,
            ScalarOp::Const(_) => false,
            ScalarOp::Dup => true,
            ScalarOp::Swap => true,
            ScalarOp::IfLtElse(_, ref a, ref b) => {
                ScalarOp::ops_require_y(&a) || ScalarOp::ops_require_y(&b)
            }
            ScalarOp::IfLeqElse(_, ref a, ref b) => {
                ScalarOp::ops_require_y(&a) || ScalarOp::ops_require_y(&b)
            }
            ScalarOp::Pow(_) => false,
            ScalarOp::Exp(_) => false,
            ScalarOp::Add(_) => false,
            ScalarOp::Sub(_) => false,
            ScalarOp::Mul(_) => false,
            ScalarOp::Div(_) => false,
            ScalarOp::Log10 => false,
        }
    }

    fn ops_require_y(ops: &[ScalarOp]) -> bool {
        for op in ops {
            if op.requires_y() {
                return true;
            }
        }
        false
    }

    fn apply_one(&self, x: &mut f64, y: &mut f64) {
        match *self {
            ScalarOp::Noop => (),
            ScalarOp::Const(v) => *x = v,
            ScalarOp::Dup => *y = *x,
            ScalarOp::Swap => mem::swap(x, y),
            ScalarOp::IfLtElse(v, ref a, ref b) => {
                *x = ScalarOp::eval(if *x < v { &a } else { &b }, *x)
            }
            ScalarOp::IfLeqElse(v, ref a, ref b) => {
                *x = ScalarOp::eval(if *x <= v { &a } else { &b }, *x)
            }
            ScalarOp::Pow(v) => *x = x.powf(v),
            ScalarOp::Exp(v) => *x = v.powf(*x),
            ScalarOp::Add(v) => *x += v,
            ScalarOp::Sub(v) => *x -= v,
            ScalarOp::Mul(v) => *x *= v,
            ScalarOp::Div(v) => *x /= v,
            ScalarOp::Log10 => *x = x.log10(),
        }
    }

    /// Evaluates a list of operations and returns the result.
    pub fn eval(ops: &[ScalarOp], x: f64) -> f64 {
        let mut x = x;
        let mut y = 0.;
        for op in ops {
            op.apply_one(&mut x, &mut y);
        }
        x
    }

    fn glsl(&self, x: &str, y: &str, ty: &str, level: u8) -> String {
        indent(level) + &match *self {
            ScalarOp::Noop => "\n".into(),
            ScalarOp::Const(v) => format!("{} = {};\n", x, format_decimal(v)),
            ScalarOp::Dup => format!("{} = {};\n", y, x),
            ScalarOp::Swap => format!("{{{0} _swap = {1}; {1} = {2}; {2} = __swap;}}\n", ty, x, y),
            ScalarOp::IfLtElse(v, ref a, ref b) => format!(
                "if ({} < {}) {{\n{}{}}} else {{\n{}{3}}}\n",
                x,
                format_decimal(v),
                ScalarOp::sub_compile(a, x, y, ty, level + 1),
                indent(level),
                ScalarOp::sub_compile(b, x, y, ty, level + 1)
            ),
            ScalarOp::IfLeqElse(v, ref a, ref b) => format!(
                "if ({} <= {}) {{\n{}{}}} else {{\n{}{3}}}\n",
                x,
                format_decimal(v),
                ScalarOp::sub_compile(a, x, y, ty, level + 1),
                indent(level),
                ScalarOp::sub_compile(b, x, y, ty, level + 1)
            ),
            ScalarOp::Pow(v) => format!("{} = pow({0}, {});\n", x, format_decimal(v)),
            ScalarOp::Exp(v) => format!("{} = pow({}, {0});\n", x, format_decimal(v)),
            ScalarOp::Add(v) => format!("{} += {};\n", x, format_decimal(v)),
            ScalarOp::Sub(v) => format!("{} -= {};\n", x, format_decimal(v)),
            ScalarOp::Mul(v) => format!("{} *= {};\n", x, format_decimal(v)),
            ScalarOp::Div(v) => format!("{} /= {};\n", x, format_decimal(v)),
            ScalarOp::Log10 => format!("{} = log({0}) / {};\n", x, 10f64.ln()),
        }
    }

    fn sub_compile(ops: &[ScalarOp], x: &str, y: &str, ty: &str, level: u8) -> String {
        let mut res = String::new();
        for op in ops {
            res += &op.glsl(x, y, ty, level);
        }
        res
    }

    /// Compiles GLSL code for the given set of operations.
    ///
    /// `var_type` should either be `float` or `double`.
    pub fn compile_shader(ops: &[ScalarOp], var_name: &str, var_type: &str) -> String {
        (if ScalarOp::ops_require_y(ops) {
            format!("{} _tmp = 0.;\n", var_type)
        } else {
            String::new()
        }) + &ScalarOp::sub_compile(ops, var_name, "_tmp", var_type, 0)
    }
}
