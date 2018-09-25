use byteorder::{ByteOrder, ReadBytesExt, BE};
use gamma::ToneCurve;
use mlu::{MLU, NO_COUNTRY, NO_LANGUAGE};
use pixel_format::MAX_CHANNELS;
use plugin::{s15fixed16_to_double, u8fixed8_to_double};
use std::any::Any;
use std::ffi::CStr;
use std::io::{self, Read, Seek};
use std::sync::Arc;
use std::mem;
use time;
use {CIExyY, CIExyYTriple, ICCTag, S15Fixed16, CIEXYZ};

pub type TypeDeserFn = fn(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>>;

/// ICC base tag
#[repr(C)]
struct TagBase {
    signature: u32,
    reserved: [u8; 4],
}

// XYZ type
// The XYZ type contains an array of three encoded values for the XYZ tristimulus values.
// Tristimulus values must be non-negative. The signed encoding allows for implementation
// optimizations by minimizing the number of fixed formats.

// -> CIEXYZ
fn xyz_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let mut buf = io::Cursor::new(buf);
    let x = s15fixed16_to_double(buf.read_i32::<BE>()?);
    let y = s15fixed16_to_double(buf.read_i32::<BE>()?);
    let z = s15fixed16_to_double(buf.read_i32::<BE>()?);
    Ok(Arc::new(CIEXYZ { x, y, z }))
}

// Chromaticity type
// The chromaticity tag type provides basic chromaticity data and type of phosphors or colorants
// of a monitor to applications and utilities.

/// -> CIExyYTriple
fn chromaticity_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let buf_len = buf.len();
    let mut buf = io::Cursor::new(buf);
    let mut channels = buf.read_u16::<BE>()?;

    // recovers from a bug introduced in early versions of lcms1
    if channels == 0 && buf_len == 32 {
        buf.read_u16::<BE>()?;
        channels = buf.read_u16::<BE>()?;
    }

    if channels != 3 {
        return Err(io::ErrorKind::InvalidData.into());
    }

    let _table = buf.read_u16::<BE>()?;
    let red_x = s15fixed16_to_double(buf.read_i32::<BE>()?);
    let red_y = s15fixed16_to_double(buf.read_i32::<BE>()?);
    let green_x = s15fixed16_to_double(buf.read_i32::<BE>()?);
    let green_y = s15fixed16_to_double(buf.read_i32::<BE>()?);
    let blue_x = s15fixed16_to_double(buf.read_i32::<BE>()?);
    let blue_y = s15fixed16_to_double(buf.read_i32::<BE>()?);

    Ok(Arc::new(CIExyYTriple {
        red: CIExyY {
            x: red_x,
            y: red_y,
            Y: 1.,
        },
        green: CIExyY {
            x: green_x,
            y: green_y,
            Y: 1.,
        },
        blue: CIExyY {
            x: blue_x,
            y: blue_y,
            Y: 1.,
        },
    }))
}

// Colorant order type
// This is an optional tag which specifies the laydown order in which colorants will be printed on
// an n-colorant device. The laydown order may be the same as the channel generation order listed
// in the colorantTableTag or the channel order of a color space such as CMYK, in which case this
// tag is not needed. When this is not the case (for example, ink-towers sometimes use the order
// KCMY), this tag may be used to specify the laydown order of the colorants.

/// -> Vec<u8>
fn colorant_order_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let mut buf = io::Cursor::new(buf);
    let count = buf.read_u32::<BE>()? as usize;
    if count > MAX_CHANNELS {
        return Err(io::ErrorKind::InvalidData.into());
    }

    let mut colorant_order = Vec::with_capacity(count);
    colorant_order.resize(count, 0);
    // (no end marker)

    buf.read_exact(&mut colorant_order)?;

    Ok(Arc::new(colorant_order))
}

// S15Fixed16 array type
// This type represents an array of generic 4-byte/32-bit fixed point quantity. The number of values
// is determined from the size of the tag.

/// -> Vec<f64>
fn s15fixed16_array_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let buf_len = buf.len();
    let mut buf = io::Cursor::new(buf);
    let count = buf_len / mem::size_of::<S15Fixed16>();

    let mut values = Vec::with_capacity(count);

    for _ in 0..count {
        values.push(s15fixed16_to_double(buf.read_i32::<BE>()?));
    }

    Ok(Arc::new(values))
}

// U16Fixed16 array type
// This type represents an array of generic 4-byte/32-bit fixed point quantity. The number of values
// is determined from the size of the tag.

/// -> Vec<f64>
fn u15fixed16_array_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let buf_len = buf.len();
    let mut buf = io::Cursor::new(buf);
    let count = buf_len / mem::size_of::<S15Fixed16>();

    let mut values = Vec::with_capacity(count);

    for _ in 0..count {
        values.push(buf.read_u32::<BE>()? as f64 / 65536.);
    }

    Ok(Arc::new(values))
}

// Signature type
// The signature type contains a four-byte sequence.
// Sequences of less than four characters are padded at the end with spaces, 20h.
// Typically this type is used for registered tags that can be displayed on many development systems
// as a sequence of four characters.

/// -> u32
fn signature_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    if buf.len() < 4 {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(Arc::new(BE::read_u32(buf)))
}

/// For some reason, bits of text often have a lot of trailing null bytes, so here’s a function that
/// returns a sub-slice with everything up to and including the first.
fn cstr_slice(buf: &[u8]) -> &[u8] {
    match buf.iter().position(|x| *x == 0) {
        Some(i) => &buf[..i + 1],
        None => buf,
    }
}

// Text type
// The text type is a simple text structure that contains a 7-bit ASCII text string. The length of
// the string is obtained by subtracting 8 from the element size portion of the tag itself. This
// string must be terminated with a 00h byte.

/// -> MLU
fn text_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let mut mlu = MLU::new();
    let cstr_buf = cstr_slice(&buf);
    let cstr = CStr::from_bytes_with_nul(cstr_buf)
        .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;
    mlu.set(NO_LANGUAGE, NO_COUNTRY, &cstr.to_string_lossy());

    Ok(Arc::new(mlu))
}

// Data type
// General-purpose data type

struct ICCData {
    flags: u32,
    data: Vec<u8>,
}

/// -> ICCData
fn data_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let data_len = buf.len() - 1;
    let mut buf = io::Cursor::new(buf);
    let mut data = Vec::with_capacity(data_len);
    let flags = buf.read_u32::<BE>()?;
    buf.read_exact(&mut data)?;

    Ok(Arc::new(ICCData { flags, data }))
}

// Text description type

/// -> MLU
fn text_description_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let mut buf = io::Cursor::new(buf);
    let mut mlu = MLU::new();

    let ascii_count = buf.read_u32::<BE>()? as usize;
    let mut cstr_buf = Vec::with_capacity(ascii_count);
    cstr_buf.resize(ascii_count, 0);
    buf.read_exact(&mut cstr_buf)?;
    let cstr = CStr::from_bytes_with_nul(cstr_slice(&cstr_buf))
        .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;
    mlu.set(NO_LANGUAGE, NO_COUNTRY, &cstr.to_string_lossy());

    // skip unicode
    let _unicode_code = buf.read_u32::<BE>()?;
    let unicode_count = buf.read_u32::<BE>()?;
    buf.seek(io::SeekFrom::Current(unicode_count as i64))?;

    // ignore scriptcode
    // (no need to dummy-read it in Rust; though this does allow for invalid data here)

    Ok(Arc::new(mlu))
}

// Curve type

/// -> ToneCurve
fn curve_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let mut buf = io::Cursor::new(buf);

    let count = buf.read_u32::<BE>()? as usize;

    match count {
        0 => {
            // linear
            Ok(Arc::new(ToneCurve::new_parametric(1, &[1.]).map_err(
                |_| io::Error::from(io::ErrorKind::InvalidData),
            )?))
        }
        1 => {
            // gamma exponent
            let gamma = u8fixed8_to_double(buf.read_u16::<BE>()?);

            Ok(Arc::new(ToneCurve::new_parametric(1, &[gamma]).map_err(
                |_| io::Error::from(io::ErrorKind::InvalidData),
            )?))
        }
        _ => {
            if count > 0x7FFF {
                return Err(io::ErrorKind::InvalidData.into());
            }

            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(buf.read_u16::<BE>()?);
            }

            Ok(Arc::new(ToneCurve::new_table(values).map_err(|_| {
                io::Error::from(io::ErrorKind::InvalidData)
            })?))
        }
    }
}

// Parametric curve type

/// -> ToneCurve
fn parametric_curve_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    const PARAMS_BY_TYPE: &[u8] = &[1, 3, 4, 5, 7];

    let mut buf = io::Cursor::new(buf);

    let p_type = buf.read_u16::<BE>()? as usize;
    let _reserved = buf.read_u16::<BE>()?;

    if p_type > 4 {
        // unknown
        return Err(io::ErrorKind::InvalidData.into());
    }

    let mut params = Vec::new();
    for _ in 0..PARAMS_BY_TYPE[p_type] {
        params.push(s15fixed16_to_double(buf.read_i32::<BE>()?));
    }

    let tone_curve = match ToneCurve::new_parametric(p_type as i32 + 1, &params) {
        Ok(curve) => curve,
        Err(_) => return Err(io::ErrorKind::InvalidData.into()),
    };

    Ok(Arc::new(tone_curve))
}

// Date time type
// A 12-byte value representation of the time and date, where the byte usage is assigned as
// specified in table 1. The actual values are encoded as 16-bit unsigned integers (u16 - see
// 5.1.6).
//
// All the dateTimeNumber values in a profile shall be in Coordinated Universal Time (UTC, also
// known as GMT or ZULU Time). Profile writers are required to convert local time to UTC when
// setting these values. Programmes that display these values may show the dateTimeNumber as UTC,
// show the equivalent local time (at current locale), or display both UTC and local versions of the
// dateTimeNumber.

/// ICC date time
#[repr(C)]
pub struct ICCDateTime {
    year: u16,
    month: u16,
    day: u16,
    hours: u16,
    minutes: u16,
    seconds: u16,
}

pub fn decode_date_time(date_time: &ICCDateTime) -> time::Tm {
    time::Tm {
        tm_nsec: 0,
        tm_sec: u16::from_be(date_time.seconds) as i32,
        tm_min: u16::from_be(date_time.minutes) as i32,
        tm_hour: u16::from_be(date_time.hours) as i32,
        tm_mday: u16::from_be(date_time.day) as i32,
        tm_mon: (u16::from_be(date_time.month) as i32) - 1,
        tm_year: (u16::from_be(date_time.year) as i32) - 1900,
        tm_wday: -1,
        tm_yday: -1,
        tm_isdst: 0,
        tm_utcoff: 0,
    }
}

fn date_time_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let mut buf = io::Cursor::new(buf);
    let mut date_time = Vec::new();
    date_time.resize(mem::size_of::<ICCDateTime>(), 0);
    buf.read_exact(&mut date_time)?;

    Ok(Arc::new(decode_date_time(unsafe {
        &*(&*date_time as *const [u8] as *const ICCDateTime)
    })))
}

// Measurement type
// The measurementType information refers only to the internal profile data and is meant to provide
// profile makers an alternative to the default measurement specifications.

fn measurement_deser(_buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    unimplemented!("deser measurement")
}

// Multilocalized unicode type
//
//   Do NOT trust SizeOfTag as there is an issue on the definition of profileSequenceDescTag. See
//   the TechNote from Max Derhak and Rohit Patil about this: basically the size of the string table
//   should be guessed and cannot be taken from the size of tag if this tag is embedded as part of
//   bigger structures (profileSequenceDescTag, for instance)
//

fn mlu_deser(buf: &[u8]) -> io::Result<Arc<Any + Send + Sync>> {
    let buf_len = buf.len();
    let mut buf = io::Cursor::new(buf);

    let count = buf.read_u32::<BE>()? as usize;
    let record_len = buf.read_u32::<BE>()?;

    if record_len != 12 {
        // not supported
        return Err(io::ErrorKind::InvalidData.into());
    }

    let mut mlu = MLU::new();

    for _ in 0..count {
        let lang = buf.read_u16::<BE>()?;
        let country = buf.read_u16::<BE>()?;
        let len = buf.read_u32::<BE>()? as usize / 2;
        let offset = buf.read_u32::<BE>()? as usize;

        if offset + len > buf_len + 8 {
            return Err(io::ErrorKind::InvalidData.into());
        }

        let pos = buf.position();

        buf.seek(io::SeekFrom::Start(offset as u64 - 8))?;

        let mut str_buf = Vec::with_capacity(len);
        for _ in 0..len {
            match buf.read_u16::<BE>()? {
                0 => break,
                c => str_buf.push(c),
            }
        }

        let string = String::from_utf16_lossy(&str_buf);
        mlu.set_raw(lang, country, string);

        buf.set_position(pos);
    }

    Ok(Arc::new(mlu))
}

// TODO: types.c:1584

use ICCTagType as Type;
const TAGS: &[(ICCTag, &[Type])] = &[
    (ICCTag::AToB0, &[Type::Lut16, Type::LutAToB, Type::Lut8]),
    (ICCTag::AToB1, &[Type::Lut16, Type::LutAToB, Type::Lut8]),
    (ICCTag::AToB2, &[Type::Lut16, Type::LutAToB, Type::Lut8]),
    (ICCTag::BToA0, &[Type::Lut16, Type::LutBToA, Type::Lut8]),
    (ICCTag::BToA1, &[Type::Lut16, Type::LutBToA, Type::Lut8]),
    (ICCTag::BToA2, &[Type::Lut16, Type::LutBToA, Type::Lut8]),
    (ICCTag::RedColorant, &[Type::XYZ]),
    (ICCTag::GreenColorant, &[Type::XYZ]),
    (ICCTag::BlueColorant, &[Type::XYZ]),
    (ICCTag::RedTRC, &[Type::Curve, Type::ParametricCurve]),
    (ICCTag::GreenTRC, &[Type::Curve, Type::ParametricCurve]),
    (ICCTag::BlueTRC, &[Type::Curve, Type::ParametricCurve]),
    (ICCTag::CalibrationDateTime, &[Type::DateTime]),
    (ICCTag::CharTarget, &[Type::Text]),
    (ICCTag::ChromaticAdaptation, &[Type::S15Fixed16Array]),
    (ICCTag::Chromaticity, &[Type::Chromaticity]),
    (ICCTag::ColorantOrder, &[Type::ColorantOrder]),
    (ICCTag::ColorantTable, &[Type::ColorantTable]),
    (ICCTag::ColorantTableOut, &[Type::ColorantTable]),
    (
        ICCTag::Copyright,
        &[Type::Text, Type::MLU, Type::TextDescription],
    ),
    (ICCTag::DateTime, &[Type::DateTime]),
    (
        ICCTag::DeviceMfgDesc,
        &[Type::TextDescription, Type::MLU, Type::Text],
    ),
    (
        ICCTag::DeviceModelDesc,
        &[Type::TextDescription, Type::MLU, Type::Text],
    ),
    (ICCTag::Gamut, &[Type::Lut16, Type::LutBToA, Type::Lut8]),
    (ICCTag::GrayTRC, &[Type::Curve, Type::ParametricCurve]),
    (ICCTag::Luminance, &[Type::XYZ]),
    (ICCTag::MediaBlackPoint, &[Type::XYZ]),
    (ICCTag::MediaWhitePoint, &[Type::XYZ]),
    (ICCTag::NamedColor2, &[Type::NamedColor2]),
    (ICCTag::Preview0, &[Type::Lut16, Type::LutBToA, Type::Lut8]),
    (ICCTag::Preview1, &[Type::Lut16, Type::LutBToA, Type::Lut8]),
    (ICCTag::Preview2, &[Type::Lut16, Type::LutBToA, Type::Lut8]),
    (
        ICCTag::ProfileDescription,
        &[Type::TextDescription, Type::MLU, Type::Text],
    ),
    (ICCTag::ProfileSequenceDesc, &[Type::ProfileSequenceDesc]),
    (ICCTag::Technology, &[Type::Signature]),
    (ICCTag::ColorimetricIntentImageState, &[Type::Signature]),
    (ICCTag::PerceptualRenderingIntentGamut, &[Type::Signature]),
    (ICCTag::SaturationRenderingIntentGamut, &[Type::Signature]),
    (ICCTag::Measurement, &[Type::Measurement]),
    (ICCTag::Ps2CRD0, &[Type::Data]),
    (ICCTag::Ps2CRD1, &[Type::Data]),
    (ICCTag::Ps2CRD2, &[Type::Data]),
    (ICCTag::Ps2CRD3, &[Type::Data]),
    (ICCTag::Ps2CSA, &[Type::Data]),
    (ICCTag::Ps2RenderingIntent, &[Type::Data]),
    (
        ICCTag::ViewingCondDesc,
        &[Type::TextDescription, Type::MLU, Type::Text],
    ),
    (ICCTag::UcrBg, &[Type::UcrBg]),
    (ICCTag::CrdInfo, &[Type::CrdInfo]),
    (ICCTag::DToB0, &[Type::MultiProcessElement]),
    (ICCTag::DToB1, &[Type::MultiProcessElement]),
    (ICCTag::DToB2, &[Type::MultiProcessElement]),
    (ICCTag::DToB3, &[Type::MultiProcessElement]),
    (ICCTag::BToD0, &[Type::MultiProcessElement]),
    (ICCTag::BToD1, &[Type::MultiProcessElement]),
    (ICCTag::BToD2, &[Type::MultiProcessElement]),
    (ICCTag::BToD3, &[Type::MultiProcessElement]),
    (ICCTag::ScreeningDesc, &[Type::TextDescription]),
    (ICCTag::ViewingConditions, &[Type::ViewingConditions]),
    (ICCTag::Screening, &[Type::Screening]),
    (ICCTag::Vcgt, &[Type::Vcgt]),
    (ICCTag::Meta, &[Type::Dict]),
    (ICCTag::ProfileSequenceId, &[Type::ProfileSequenceId]),
    (ICCTag::ProfileDescriptionML, &[Type::MLU]),
    (ICCTag::ArgyllArts, &[Type::S15Fixed16Array]),
];

pub fn tag_types(for_tag: ICCTag) -> Option<&'static [Type]> {
    for (tag, types) in TAGS {
        if *tag == for_tag {
            return Some(types);
        }
    }
    None
}

pub fn tag_deserializer(for_type: Type) -> Option<TypeDeserFn> {
    // TODO
    match for_type {
        Type::Chromaticity => Some(chromaticity_deser),
        Type::ColorantOrder => Some(colorant_order_deser),
        Type::ColorantTable => None,
        Type::CrdInfo => None,
        Type::Curve => Some(curve_deser),
        Type::Data => Some(data_deser),
        Type::Dict => None,
        Type::DateTime => Some(date_time_deser),
        Type::DeviceSettings => None,
        Type::Lut16 => None,
        Type::Lut8 => None,
        Type::LutAToB => None,
        Type::LutBToA => None,
        Type::Measurement => None,
        Type::MLU => Some(mlu_deser),
        Type::MultiProcessElement => None,
        Type::NamedColor => None,
        Type::NamedColor2 => None,
        Type::ParametricCurve => Some(parametric_curve_deser),
        Type::ProfileSequenceDesc => None,
        Type::ProfileSequenceId => None,
        Type::ResponseCurveSet16 => None,
        Type::S15Fixed16Array => Some(s15fixed16_array_deser),
        Type::Screening => None,
        Type::Signature => Some(signature_deser),
        Type::Text => Some(text_deser),
        Type::TextDescription => Some(text_description_deser),
        Type::U16Fixed16Array => None,
        Type::UcrBg => None,
        Type::UInt16Array => None,
        Type::UInt32Array => None,
        Type::UInt64Array => None,
        Type::UInt8Array => None,
        Type::Vcgt => None,
        Type::ViewingConditions => None,
        Type::XYZ => Some(xyz_deser),
    }
}
