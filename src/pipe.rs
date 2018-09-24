//! Transform pipeline.

use alpha::MAX_CHANNELS;
use gamma::ToneCurve;
use internal::quick_saturate_word;
use named::NamedColorList;
use pcs::{lab_to_xyz, xyz_to_lab, MAX_ENCODEABLE_XYZ};
use std::fmt;
use {CIELab, CIEXYZ};

type StageEvalFn = fn(&[f32], &mut [f32], &Stage);

/// Multi-process element types.
///
/// All types marked “impl” are actual stage types with different implementations and should all be
/// covered e.g. when translating to GPU shaders.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageType {
    /// `cvst` (**impl**)
    ///
    /// This stage type has [StageData::Curves] data.
    ///
    /// In pseudocode:
    /// ```text
    /// output_channels = input_channels.map(|value, index| {
    ///     data.curves[index].eval(value)
    /// });
    /// ```
    CurveSet = 0x63767374,

    /// `matf` (**impl**)
    ///
    /// This stage type has [StageData::Matrix] data.
    ///
    /// In pseudocode (column-major):
    /// ```text
    /// for i in 0..output_channels.len() {
    ///     let mut a = 0.;
    ///     input_channels.for_each(|value, j| {
    ///         a += value * matrix[i][j];
    ///     });
    ///     output_channels[i] = a + offsets[i];
    /// }
    /// ```
    Matrix = 0x6D617466,
    /// `clut`
    CLut = 0x636C7574,

    /// `bACS`
    BAcs = 0x62414353,
    /// `eACS`
    EAcs = 0x65414353,

    /// (non-ICC) `l2x ` (**impl**)
    ///
    /// Converts between PCS; 3 inputs & 3 outputs.
    ///
    /// (See `evaluate_xyz_to_lab` in this file for reference)
    XYZ2Lab = 0x6C327820,

    /// (non-ICC) `x2l ` (**impl**)
    ///
    /// Converts between PCS; 3 inputs & 3 outputs.
    ///
    /// (See `evaluate_lab_to_xyz` in this file for reference)
    Lab2XYZ = 0x78326C20,
    /// (non-ICC) `ncl `
    NamedColor = 0x6E636C20,
    /// (non-ICC) `2 4 `
    LabV2toV4 = 0x32203420,
    /// (non-ICC) `4 2 `
    LabV4toV2 = 0x34203220,

    /// (non-ICC) `idn ` (**impl**)
    ///
    /// Copies input to output.
    Identity = 0x69646E20,

    // Float to floatPCS
    /// (non-ICC) `d2l `
    Lab2FloatPCS = 0x64326C20,
    /// (non-ICC) `l2d `
    FloatPCS2Lab = 0x6C326420,
    /// (non-ICC) `d2x `
    XYZ2FloatPCS = 0x64327820,
    /// (non-ICC) `x2d `
    FloatPCS2XYZ = 0x78326420,

    /// (non-ICC) `clp ` (**impl**)
    ///
    /// Copies input to output and clamps all values below zero to zero.
    ClipNegatives = 0x636c7020,
}

/// Parameters for pipeline stages.
#[derive(Debug, Clone)]
pub enum StageData {
    None,
    Matrix {
        matrix: Vec<f64>,
        offset: Option<Vec<f64>>,
    },
    Curves(Vec<ToneCurve>),
    NamedColorList(NamedColorList),
}

/// A pipeline stage.
#[derive(Clone)]
pub struct Stage {
    ty: StageType,
    implements: StageType,

    input_channels: usize,
    output_channels: usize,

    eval_fn: StageEvalFn,

    data: StageData,
}

impl Stage {
    pub(crate) fn alloc(
        ty: StageType,
        input_channels: usize,
        output_channels: usize,
        eval_fn: StageEvalFn,
        data: StageData,
    ) -> Stage {
        Stage {
            ty,
            implements: ty,
            input_channels,
            output_channels,
            eval_fn,
            data,
        }
    }

    /// Creates a new tone curve stage. The number of tone curves must match the channel count.
    pub fn new_tone_curves(channels: usize, curves: &[ToneCurve]) -> Stage {
        Self::new_tone_curves_impl(channels, Some(curves))
    }

    /// Creates a bunch of identity curves.
    pub fn new_identity_curves(channels: usize) -> Stage {
        let mut stage = Stage::new_tone_curves_impl(channels, None);
        stage.implements = StageType::Identity;
        stage
    }

    fn new_tone_curves_impl(channels: usize, curves: Option<&[ToneCurve]>) -> Stage {
        let curves = if let Some(curves) = curves {
            curves.to_vec()
        } else {
            let mut curves = Vec::new();
            for _ in 0..channels {
                curves.push(ToneCurve::new_gamma(1.).unwrap());
            }
            curves
        };

        assert_eq!(curves.len(), channels);

        Stage::alloc(
            StageType::CurveSet,
            channels,
            channels,
            evaluate_curves,
            StageData::Curves(curves),
        )
    }

    /// Creates a new matrix stage. Rows and columns are input and output channels, respectively
    /// (in column-major order).
    ///
    /// Note that this should be a 3x3 matrix and that the offset shoul simply be a 3-vector.
    pub fn new_matrix(rows: usize, cols: usize, matrix: &[f64], offset: Option<&[f64]>) -> Stage {
        Self::alloc(
            StageType::Matrix,
            cols,
            rows,
            evaluate_matrix,
            StageData::Matrix {
                matrix: matrix.to_vec(),
                offset: offset.map(|x| x.to_vec()),
            },
        )
    }

    /// Creates a new stage that converts Lab V2 to Lab V4.
    pub fn new_labv2_to_v4() -> Stage {
        const V2_TO_V4: [f64; 9] = [
            65535. / 65280.,
            0.,
            0.,
            0.,
            65535. / 65280.,
            0.,
            0.,
            0.,
            65535. / 65280.,
        ];

        let mut stage = Self::new_matrix(3, 3, &V2_TO_V4, None);
        stage.implements = StageType::LabV2toV4;

        stage
    }

    /// Creates a new stage that converts Lab V4 to Lab V2.
    pub fn new_labv4_to_v2() -> Stage {
        const V4_TO_V2: [f64; 9] = [
            65280. / 65535.,
            0.,
            0.,
            0.,
            65280. / 65535.,
            0.,
            0.,
            0.,
            65280. / 65535.,
        ];

        let mut stage = Self::new_matrix(3, 3, &V4_TO_V2, None);
        stage.implements = StageType::LabV4toV2;

        stage
    }

    /// Creates a new stage that converts XYZ to Lab.
    pub fn new_xyz_to_lab() -> Stage {
        Self::alloc(
            StageType::XYZ2Lab,
            3,
            3,
            evaluate_xyz_to_lab,
            StageData::None,
        )
    }

    /// Creates a new stage that converts Lab to XYZ.
    pub fn new_lab_to_xyz() -> Stage {
        Self::alloc(
            StageType::Lab2XYZ,
            3,
            3,
            evaluate_lab_to_xyz,
            StageData::None,
        )
    }

    /// Creates a new stage that clamps all negative values to zero.
    pub fn new_clip_negatives(channels: usize) -> Stage {
        Self::alloc(
            StageType::ClipNegatives,
            channels,
            channels,
            clipper,
            StageData::None,
        )
    }

    /// Converts from Lab to float.
    ///
    /// Note that the MPE gives numbers in the normal Lab range and we need the 0..1.0 range for
    /// the formatters.
    ///
    /// ```text
    /// L*:   0...100 => 0...1.0  (L* / 100)
    /// ab*: -128..+127 to 0..1   ((ab* + 128) / 255)
    /// ```
    pub fn new_normalize_from_lab_float() -> Stage {
        const A1: [f64; 9] = [1. / 100., 0., 0., 0., 1. / 255., 0., 0., 0., 1. / 255.];

        const O1: [f64; 3] = [0., 128. / 255., 128. / 255.];

        let mut stage = Self::new_matrix(3, 3, &A1, Some(&O1));
        stage.implements = StageType::Lab2FloatPCS;
        stage
    }

    /// Converts from XYZ to floating point PCS.
    pub fn new_normalize_from_xyz_float() -> Stage {
        const A1: [f64; 9] = [
            32768. / 65535.,
            0.,
            0.,
            0.,
            32768. / 65535.,
            0.,
            0.,
            0.,
            32768. / 65535.,
        ];

        let mut stage = Self::new_matrix(3, 3, &A1, None);
        stage.implements = StageType::XYZ2FloatPCS;
        stage
    }

    pub fn new_normalize_to_lab_float() -> Stage {
        const A1: [f64; 9] = [100., 0., 0., 0., 255., 0., 0., 0., 255.];

        const O1: [f64; 3] = [0., -128., -128.];

        let mut stage = Self::new_matrix(3, 3, &A1, Some(&O1));
        stage.implements = StageType::FloatPCS2Lab;
        stage
    }

    pub fn new_normalize_to_xyz_float() -> Stage {
        const A1: [f64; 9] = [
            65535. / 32768.,
            0.,
            0.,
            0.,
            65535. / 32768.,
            0.,
            0.,
            0.,
            65535. / 32768.,
        ];

        let mut stage = Self::new_matrix(3, 3, &A1, None);
        stage.implements = StageType::FloatPCS2XYZ;
        stage
    }

    /// Creates a new identity stage, i.e. it simply copies input to output.
    pub fn new_identity(channels: usize) -> Stage {
        Stage::alloc(
            StageType::Identity,
            channels,
            channels,
            evaluate_identity,
            StageData::None,
        )
    }
}

impl Stage {
    /// The actual stage type.
    pub fn stage_type(&self) -> StageType {
        self.ty
    }

    /// The conversion kind this stage implements.
    pub fn impl_type(&self) -> StageType {
        self.implements
    }

    /// The number of input channels.
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// The number of output channels.
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    /// Stage parameters.
    pub fn data(&self) -> &StageData {
        &self.data
    }
}

impl fmt::Debug for Stage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Stage {{ type: {:?}, impl: {:?}, channels: {} -> {}, data: {:?} }}",
            self.ty, self.implements, self.input_channels, self.output_channels, self.data
        )
    }
}

/// Pipeline evaluator (in words)
type PipelineEval16Fn = fn(&[u16], &mut [u16], &Pipeline);

/// Pipeline evaluator (in floating point)
type PipelineEvalFloatFn = fn(&[f32], &mut [f32], &Pipeline);

/// A pipeline.
#[derive(Clone)]
pub struct Pipeline {
    elements: Vec<Stage>,
    input_channels: usize,
    output_channels: usize,

    eval_16_fn: PipelineEval16Fn,
    eval_float_fn: PipelineEvalFloatFn,
}

impl Pipeline {
    /// Creates a new empty pipeline. Must have fewer than 17 channels.
    ///
    /// # Panics
    /// Will panic if there are too many channels.
    pub fn new(input_channels: usize, output_channels: usize) -> Pipeline {
        if input_channels >= MAX_CHANNELS || output_channels >= MAX_CHANNELS {
            panic!("Pipeline has too many channels");
        }

        let mut lut = Pipeline {
            elements: Vec::new(),
            input_channels,
            output_channels,
            eval_16_fn: lut_eval_16,
            eval_float_fn: lut_eval_float,
        };

        lut.update_channels();

        lut
    }

    /// The number of input channels.
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// The number of output channels.
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    fn update_channels(&mut self) {
        // We can set the input/output channels only if we have elements.
        if !self.elements.is_empty() {
            let first = self.elements.first().unwrap();
            let last = self.elements.last().unwrap();

            self.input_channels = first.input_channels;
            self.output_channels = last.output_channels;

            // don’t need to check chain consistency
        }
    }

    /// Pipeline stages, in order.
    pub fn stages(&self) -> &[Stage] {
        &self.elements
    }

    /// Evaluates the pipeline with u16.
    pub fn eval_16(&self, input: &[u16], output: &mut [u16]) {
        (self.eval_16_fn)(input, output, self);
    }

    /// Evaluates the pipeline with floats.
    pub fn eval_float(&self, input: &[f32], output: &mut [f32]) {
        (self.eval_float_fn)(input, output, self);
    }

    pub(crate) fn prepend_stage(&mut self, stage: Stage) {
        self.elements.insert(0, stage);
        self.update_channels();
    }

    pub(crate) fn append_stage(&mut self, stage: Stage) {
        self.elements.push(stage);
        self.update_channels();
    }

    /// Concatenates two LUT into a new single one
    pub(crate) fn concat(&mut self, other: &Pipeline) {
        // If both LUTS does not have elements, we need to inherit
        // the number of channels
        if self.elements.is_empty() && other.elements.is_empty() {
            self.input_channels = other.input_channels;
            self.output_channels = other.output_channels;
        }

        // Cat second
        for stage in &other.elements {
            self.elements.push(stage.clone());
        }

        self.update_channels();
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Pipeline {{ elements: {:?}, channels: {} -> {} }}",
            self.elements, self.input_channels, self.output_channels
        )
    }
}

/// From floating point to 16 bits
fn from_float_to_16(input: &[f32], output: &mut [u16], n: usize) {
    for i in 0..n {
        output[i] = quick_saturate_word((input[i] * 65535.).into());
    }
}

/// From 16 bits to floating point
fn from_16_to_float(input: &[u16], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i] as f32 / 65535.;
    }
}

fn copy_float_slice(src: &[f32], dest: &mut [f32]) {
    if src.len() > dest.len() {
        let dest_len = dest.len();
        dest.copy_from_slice(&src[0..dest_len]);
    } else {
        dest[0..src.len()].copy_from_slice(src);
    }
}

const MAX_STAGE_CHANNELS: usize = 128;

/// Default function for evaluating the LUT with 16 bits. Precision is retained.
fn lut_eval_16(input: &[u16], output: &mut [u16], pipeline: &Pipeline) {
    let mut phase = 0;
    let mut storage = [[0.; MAX_STAGE_CHANNELS], [0.; MAX_STAGE_CHANNELS]];
    from_16_to_float(input, &mut storage[phase]);

    for stage in &pipeline.elements {
        let next_phase = phase ^ 1;
        let next_phase_yes_this_is_safe =
            unsafe { &mut *(&storage[next_phase] as *const _ as *mut [f32; MAX_STAGE_CHANNELS]) };
        (stage.eval_fn)(&storage[phase], next_phase_yes_this_is_safe, &stage);
        phase = next_phase;
    }

    from_float_to_16(&storage[phase], output, pipeline.output_channels as usize);
}

/// Evaluates the LUt with floats.
fn lut_eval_float(input: &[f32], output: &mut [f32], pipeline: &Pipeline) {
    let mut phase = 0;
    let mut storage = [[0.; MAX_STAGE_CHANNELS], [0.; MAX_STAGE_CHANNELS]];

    copy_float_slice(input, &mut storage[phase]);

    for stage in &pipeline.elements {
        let next_phase = phase ^ 1;
        let next_phase_yes_this_is_safe =
            unsafe { &mut *(&storage[next_phase] as *const _ as *mut [f32; MAX_STAGE_CHANNELS]) };
        (stage.eval_fn)(&storage[phase], next_phase_yes_this_is_safe, &stage);
        phase = next_phase;
    }

    copy_float_slice(&storage[phase], output);
}

/// Special care should be taken here because precision loss. A temporary cmsFloat64Number buffer is being used
fn evaluate_matrix(input: &[f32], output: &mut [f32], stage: &Stage) {
    let (matrix, offset) = match stage.data {
        StageData::Matrix {
            ref matrix,
            ref offset,
        } => (matrix, offset),
        _ => panic!("Invalid stage data (this shouldn’t happen)"),
    };

    // Input is already in 0..1.0 notation
    for i in 0..stage.output_channels {
        let mut tmp = 0.;
        for j in 0..stage.input_channels {
            tmp += input[j as usize] as f64 * matrix[(i * stage.input_channels + j) as usize];
        }
        if let Some(offset) = offset {
            tmp += offset[i as usize];
        }
        output[i as usize] = tmp as f32;
    }
    // Output in 0..1.0 domain
}

fn evaluate_curves(input: &[f32], output: &mut [f32], stage: &Stage) {
    let curves = match stage.data {
        StageData::Curves(ref c) => c,
        _ => panic!("Invalid stage data (this shouldn’t happen)"),
    };

    for i in 0..curves.len() {
        output[i] = curves[i].eval_float(input[i]);
    }
}

fn evaluate_xyz_to_lab(input: &[f32], output: &mut [f32], _: &Stage) {
    // From 0..1.0 to XYZ
    let xyz = CIEXYZ {
        x: input[0] as f64 * MAX_ENCODEABLE_XYZ,
        y: input[1] as f64 * MAX_ENCODEABLE_XYZ,
        z: input[2] as f64 * MAX_ENCODEABLE_XYZ,
    };

    let lab = xyz_to_lab(None, xyz);

    // From V4 Lab to 0..1.0
    output[0] = (lab.L / 100.) as f32;
    output[1] = ((lab.a + 128.) / 255.) as f32;
    output[2] = ((lab.b + 128.) / 255.) as f32;
}

fn evaluate_lab_to_xyz(input: &[f32], output: &mut [f32], _: &Stage) {
    // V4 rules
    let lab = CIELab {
        L: input[0] as f64 * 100.,
        a: input[1] as f64 * 255. - 128.,
        b: input[2] as f64 * 255. - 128.,
    };

    let xyz = lab_to_xyz(None, lab);

    // From XYZ, range 0..19997 to 0..1.0, note that 1.99997 comes from 0xffff
    // encoded as 1.15 fixed point, so 1 + (32767.0 / 32768.0)

    output[0] = (xyz.x / MAX_ENCODEABLE_XYZ) as f32;
    output[1] = (xyz.y / MAX_ENCODEABLE_XYZ) as f32;
    output[2] = (xyz.z / MAX_ENCODEABLE_XYZ) as f32;
}

/// Clips values smaller than zero
fn clipper(input: &[f32], output: &mut [f32], stage: &Stage) {
    for i in 0..stage.input_channels as usize {
        output[i] = input[i].max(0.);
    }
}

fn evaluate_identity(input: &[f32], output: &mut [f32], _: &Stage) {
    copy_float_slice(input, output);
}
