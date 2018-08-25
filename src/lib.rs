//! Color management system.

// temporary:
#![allow(dead_code)]

extern crate cgmath;
extern crate libc;
extern crate time;
#[macro_use]
extern crate bitflags;

#[macro_use]
mod macros;
mod alpha;
mod convert;
pub mod gamma;
mod gamut;
mod half;
mod internal;
mod lut;
pub mod mlu;
mod named;
pub mod op;
mod optimization;
mod pack;
pub mod pcs;
pub mod pixel_format;
mod plugin;
pub mod profile;
mod sampling;
pub mod transform;
pub mod transform_tmp;
mod virtuals;
pub mod white_point;

pub use gamma::ToneCurve;
pub use profile::Profile;
pub use transform::Transform;

#[cfg(test)]
mod tests;

pub type S15Fixed16 = i32;
pub type PixelFormat = u32;

/// Rendering intents.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Intent {
    // ICC intents
    Perceptual = 0,
    RelativeColorimetric = 1,
    Saturation = 2,
    AbsoluteColorimetric = 3,

    // Non-ICC intents
    PreserveKOnlyPerceptual = 10,
    PreserveKOnlyRelativeColorimetric = 11,
    PreserveKOnlySaturation = 12,
    PreserveKPlanePerceptual = 13,
    PreserveKPlaneRelativeColorimetric = 14,
    PreserveKPlaneSaturation = 15,
}

/// ICC color spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// `XYZ `
    XYZ = 0x58595a20,
    /// `Lab `
    Lab = 0x4C616220,
    /// `Luv `
    Luv = 0x4C757620,
    /// `YCbr`
    YCbCr = 0x59436272,
    /// `Yxy `
    Yxy = 0x59787920,
    /// `RGB `
    RGB = 0x52474220,
    /// `GRAY`
    Gray = 0x47524159,
    /// `HSV `
    HSV = 0x48535620,
    /// `HLS `
    HLS = 0x484C5320,
    /// `CMYK`
    CMYK = 0x434D594B,
    /// `CMY `
    CMY = 0x434D5920,
    /// `MCH1`
    MCH1 = 0x4D434831,
    /// `MCH2`
    MCH2 = 0x4D434832,
    /// `MCH3`
    MCH3 = 0x4D434833,
    /// `MCH4`
    MCH4 = 0x4D434834,
    /// `MCH5`
    MCH5 = 0x4D434835,
    /// `MCH6`
    MCH6 = 0x4D434836,
    /// `MCH7`
    MCH7 = 0x4D434837,
    /// `MCH8`
    MCH8 = 0x4D434838,
    /// `MCH9`
    MCH9 = 0x4D434839,
    /// `MCHA`
    MCHA = 0x4D434841,
    /// `MCHB`
    MCHB = 0x4D434842,
    /// `MCHC`
    MCHC = 0x4D434843,
    /// `MCHD`
    MCHD = 0x4D434844,
    /// `MCHE`
    MCHE = 0x4D434845,
    /// `MCHF`
    MCHF = 0x4D434846,
    /// `nmcl`
    Named = 0x6E6D636C,
    /// `1CLR`
    S1Color = 0x31434C52,
    /// `2CLR`
    S2Color = 0x32434C52,
    /// `3CLR`
    S3Color = 0x33434C52,
    /// `4CLR`
    S4Color = 0x34434C52,
    /// `5CLR`
    S5Color = 0x35434C52,
    /// `6CLR`
    S6Color = 0x36434C52,
    /// `7CLR`
    S7Color = 0x37434C52,
    /// `8CLR`
    S8Color = 0x38434C52,
    /// `9CLR`
    S9Color = 0x39434C52,
    /// `ACLR`
    S10Color = 0x41434C52,
    /// `BCLR`
    S11Color = 0x42434C52,
    /// `CCLR`
    S12Color = 0x43434C52,
    /// `DCLR`
    S13Color = 0x44434C52,
    /// `ECLR`
    S14Color = 0x45434C52,
    /// `FCLR`
    S15Color = 0x46434C52,
    /// `LuvK`
    LuvK = 0x4C75764B,
}

/// Profile connection spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PCS {
    XYZ,
    Lab,
}

impl ColorSpace {
    /// Returns the number of channels in the color space.
    pub fn channels(self) -> usize {
        use ColorSpace::*;
        match self {
            MCH1 | S1Color | Gray => 1,
            MCH2 | S2Color => 2,
            XYZ | Lab | Luv | YCbCr | Yxy | RGB | HSV | HLS | CMY | MCH3 | S3Color => 3,
            LuvK | CMYK | MCH4 | S4Color => 4,
            MCH5 | S5Color => 5,
            MCH6 | S6Color => 6,
            MCH7 | S7Color => 7,
            MCH8 | S8Color => 8,
            MCH9 | S9Color => 9,
            MCHA | S10Color => 10,
            MCHB | S11Color => 11,
            MCHC | S12Color => 12,
            MCHD | S13Color => 13,
            MCHE | S14Color => 14,
            MCHF | S15Color => 15,
            Named => 0, // ?
        }
    }
}

/// Pixel types.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelType {
    /// Don’t check color space
    Any = 0,
    // 1 & 2 are reserved
    Gray = 3,
    RGB = 4,
    CMY = 5,
    CMYK = 6,
    YCbCr = 7,
    /// Lu'v'
    YUV = 8,
    XYZ = 9,
    Lab = 10,
    /// Lu'v'K
    YUVK = 11,
    HSV = 12,
    HLS = 13,
    Yxy = 14,

    MCH1 = 15,
    MCH2 = 16,
    MCH3 = 17,
    MCH4 = 18,
    MCH5 = 19,
    MCH6 = 20,
    MCH7 = 21,
    MCH8 = 22,
    MCH9 = 23,
    MCH10 = 24,
    MCH11 = 25,
    MCH12 = 26,
    MCH13 = 27,
    MCH14 = 28,
    MCH15 = 29,

    // Identical to PixelType::Lab, but using the V2 old encoding
    LabV2 = 30,
}

/// Pixel types.
/// Base ICC tag definitions.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ICCTag {
    /// `A2B0`
    AToB0 = 0x41324230,
    /// `A2B1`
    AToB1 = 0x41324231,
    /// `A2B2`
    AToB2 = 0x41324232,
    /// `bXYZ`
    BlueColorant = 0x6258595A,
    /// `bTRC`
    BlueTRC = 0x62545243,
    /// `B2A0`
    BToA0 = 0x42324130,
    /// `B2A1`
    BToA1 = 0x42324131,
    /// `B2A2`
    BToA2 = 0x42324132,
    /// `calt`
    CalibrationDateTime = 0x63616C74,
    /// `targ`
    CharTarget = 0x74617267,
    /// `chad`
    ChromaticAdaptation = 0x63686164,
    /// `chrm`
    Chromaticity = 0x6368726D,
    /// `clro`
    ColorantOrder = 0x636C726F,
    /// `clrt`
    ColorantTable = 0x636C7274,
    /// `clot`
    ColorantTableOut = 0x636C6F74,
    /// `ciis`
    ColorimetricIntentImageState = 0x63696973,
    /// `cprt`
    Copyright = 0x63707274,
    /// `crdi`
    CrdInfo = 0x63726469,
    /// `data`
    Data = 0x64617461,
    /// `dtim`
    DateTime = 0x6474696D,
    /// `dmnd`
    DeviceMfgDesc = 0x646D6E64,
    /// `dmdd`
    DeviceModelDesc = 0x646D6464,
    /// `devs`
    DeviceSettings = 0x64657673,
    /// `D2B0`
    DToB0 = 0x44324230,
    /// `D2B1`
    DToB1 = 0x44324231,
    /// `D2B2`
    DToB2 = 0x44324232,
    /// `D2B3`
    DToB3 = 0x44324233,
    /// `B2D0`
    BToD0 = 0x42324430,
    /// `B2D1`
    BToD1 = 0x42324431,
    /// `B2D2`
    BToD2 = 0x42324432,
    /// `B2D3`
    BToD3 = 0x42324433,
    /// `gamt`
    Gamut = 0x67616D74,
    /// `kTRC`
    GrayTRC = 0x6b545243,
    /// `gXYZ`
    GreenColorant = 0x6758595A,
    /// `gTRC`
    GreenTRC = 0x67545243,
    /// `lumi`
    Luminance = 0x6C756d69,
    /// `meas`
    Measurement = 0x6D656173,
    /// `bkpt`
    MediaBlackPoint = 0x626B7074,
    /// `wtpt`
    MediaWhitePoint = 0x77747074,
    /// `ncol` (deprecated)
    NamedColor = 0x6E636f6C,
    /// `ncl2`
    NamedColor2 = 0x6E636C32,
    /// `resp`
    OutputResponse = 0x72657370,
    /// `rig0`
    PerceptualRenderingIntentGamut = 0x72696730,
    /// `pre0`
    Preview0 = 0x70726530,
    /// `pre1`
    Preview1 = 0x70726531,
    /// `pre2`
    Preview2 = 0x70726532,
    /// `desc`
    ProfileDescription = 0x64657363,
    /// `dscm`
    ProfileDescriptionML = 0x6473636d,
    /// `pseq`
    ProfileSequenceDesc = 0x70736571,
    /// `psid`
    ProfileSequenceId = 0x70736964,
    /// `psd0`
    Ps2CRD0 = 0x70736430,
    /// `psd1`
    Ps2CRD1 = 0x70736431,
    /// `psd2`
    Ps2CRD2 = 0x70736432,
    /// `psd3`
    Ps2CRD3 = 0x70736433,
    /// `ps2s`
    Ps2CSA = 0x70733273,
    /// `ps2i`
    Ps2RenderingIntent = 0x70733269,
    /// `rXYZ`
    RedColorant = 0x7258595A,
    /// `rTRC`
    RedTRC = 0x72545243,
    /// `rig2`
    SaturationRenderingIntentGamut = 0x72696732,
    /// `scrd`
    ScreeningDesc = 0x73637264,
    /// `scrn`
    Screening = 0x7363726E,
    /// `tech`
    Technology = 0x74656368,
    /// `bfd `
    UcrBg = 0x62666420,
    /// `vued`
    ViewingCondDesc = 0x76756564,
    /// `view`
    ViewingConditions = 0x76696577,
    /// `vcgt`
    Vcgt = 0x76636774,
    /// `meta`
    Meta = 0x6D657461,
    /// `arts`
    ArgyllArts = 0x61727473,
    /* BlueMatrixColumnTag = 0x6258595A, // 'bXYZ'
     * GreenMatrixColumnTag = 0x6758595A, // 'gXYZ'
     * RedMatrixColumnTag = 0x7258595A, // 'rXYZ' */
}

/// ICC profile classes.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfileClass {
    /// `scnr`
    Input = 0x73636E72,
    /// `mntr`
    Display = 0x6D6E7472,
    /// `prtr`
    Output = 0x70727472,
    /// `link`
    Link = 0x6C696E6B,
    /// `abst`
    Abstract = 0x61627374,
    /// `spac`
    ColorSpace = 0x73706163,
    /// `nmcl`
    NamedColor = 0x6e6d636c,
}

/// A color in the XYZ color space.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CIEXYZ {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Into<cgmath::Vector3<f64>> for CIEXYZ {
    fn into(self) -> cgmath::Vector3<f64> {
        cgmath::Vector3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

/// A color in the xyY color space.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_snake_case)]
pub struct CIExyY {
    pub x: f64,
    pub y: f64,
    pub Y: f64,
}

/// A color in the CIE L* a* b* color space.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_snake_case)]
pub struct CIELab {
    pub L: f64,
    pub a: f64,
    pub b: f64,
}

/// A color in the CIE L* c* h color space.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_snake_case)]
pub struct CIELCh {
    pub L: f64,
    pub C: f64,
    pub h: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_snake_case)]
pub struct JCh {
    pub J: f64,
    pub C: f64,
    pub h: f64,
}

/// An XYZ triple, representing RGB primaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CIEXYZTriple {
    pub red: CIEXYZ,
    pub green: CIEXYZ,
    pub blue: CIEXYZ,
}

/// An xyY triple, representing RGB primaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CIExyYTriple {
    pub red: CIExyY,
    pub green: CIExyY,
    pub blue: CIExyY,
}

/// ICC Technology tag.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Technology {
    /// `dcam`
    DigitalCamera = 0x6463616D,
    /// `fscn`
    FilmScanner = 0x6673636E,
    /// `rscn`
    ReflectiveScanner = 0x7273636E,
    /// `ijet`
    InkJetPrinter = 0x696A6574,
    /// `twax`
    ThermalWaxPrinter = 0x74776178,
    /// `epho`
    ElectrophotographicPrinter = 0x6570686F,
    /// `esta`
    ElectrostaticPrinter = 0x65737461,
    /// `dsub`
    DyeSublimationPrinter = 0x64737562,
    /// `rpho`
    PhotographicPaperPrinter = 0x7270686F,
    /// `fprn`
    FilmWriter = 0x6670726E,
    /// `vidm`
    VideoMonitor = 0x7669646D,
    /// `vidc`
    VideoCamera = 0x76696463,
    /// `pjtv`
    ProjectionTelevision = 0x706A7476,
    /// `CRT `
    CRTDisplay = 0x43525420,
    /// `PMD `
    PMDisplay = 0x504D4420,
    /// `AMD `
    AMDisplay = 0x414D4420,
    /// `KPCD`
    PhotoCD = 0x4B504344,
    /// `imgs`
    PhotoImageSetter = 0x696D6773,
    /// `grav`
    Gravure = 0x67726176,
    /// `offs`
    OffsetLithography = 0x6F666673,
    /// `silk`
    Silkscreen = 0x73696C6B,
    /// `flex`
    Flexography = 0x666C6578,
    /// `mpfs`
    MotionPictureFilmScanner = 0x6D706673,
    /// `mpfr`
    MotionPictureFilmRecorder = 0x6D706672,
    /// `dmpc`
    DigitalMotionPictureCamera = 0x646D7063,
    /// `dcpj`
    DigitalCinemaProjector = 0x64636A70,
}
