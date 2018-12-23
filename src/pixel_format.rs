//! Pixel format encoding and decoding.

use crate::internal::quick_saturate_word;
use crate::ColorSpace;
use std::any::Any;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

// FIXME: static conversion fns are invalid because DynPixelFormats are mutable

/// Maximum number of channels per pixel.
pub const MAX_CHANNELS: usize = 16;

/// Maximum number of bytes per pixel.
///
/// No pixel format should require more elements per pixel than this, as doing so results in undefined behavior.
pub const MAX_BYTES_PER_PIXEL: usize = 256;

/// A function that decodes a single pixel at index 0 of the given array slice and puts the output in the u16 array.
pub type Decode16Fn<T> = fn(&[T], &mut [u16; MAX_CHANNELS]);

/// A function that decodes a single pixel at index 0 of the given array slice and puts the output in the f32 array.
pub type DecodeFloatFn<T> = fn(&[T], &mut [f32; MAX_CHANNELS]);

/// A function that encodes a single pixel to index 0 of the given array slice.
pub type Encode16Fn<T> = fn(&[u16; MAX_CHANNELS], &mut [T]);

/// A function that encodes a single pixel to index 0 of the given array slice.
pub type EncodeFloatFn<T> = fn(&[f32; MAX_CHANNELS], &mut [T]);

/// Very unsafe.
struct StaticPixelFormatInfo {
    decode_16_fn: fn(&[u8], &mut [u16; MAX_CHANNELS]),
    decode_float_fn: fn(&[u8], &mut [f32; MAX_CHANNELS]),
    encode_16_fn: fn(&[u16; MAX_CHANNELS], &mut [u8]),
    encode_float_fn: fn(&[f32; MAX_CHANNELS], &mut [u8]),
}

/// A static pixel format. (Also see [DynPixelFormat])
pub trait PixelFormat {
    /// The primitive type of the pixel format.
    ///
    /// For example, 8-bit image formats would use u8.
    type Element: Copy;

    /// The pixel format’s color space.
    const SPACE: ColorSpace;

    /// If true, the pixel format is a floating-point format.
    ///
    /// This is used for deciding whether the transform should be performed in floats.
    const IS_FLOAT: bool;

    /// The channel count.
    const CHANNELS: usize;

    /// The number of extra channels (such as alpha).
    const EXTRA_CHANNELS: usize;

    /// If true, the pixel format has a layout where extra channels come first.
    const EXTRA_FIRST: bool;

    /// If true, the order of the color channel elements will be reversed.
    const REVERSE: bool;

    /// Returns the 16-bit decoder function.
    const DECODE_16_FN: Decode16Fn<Self::Element>;

    /// Returns the floating-point decoder function.
    const DECODE_FLOAT_FN: DecodeFloatFn<Self::Element>;

    /// Returns the 16-bit encoder function.
    const ENCODE_16_FN: Encode16Fn<Self::Element>;

    /// Returns the floating-point encoder function.
    const ENCODE_FLOAT_FN: EncodeFloatFn<Self::Element>;

    /// Returns the number of total channels (i.e. element units per pixel).
    fn total_channels() -> usize {
        Self::CHANNELS + Self::EXTRA_CHANNELS
    }

    /// Returns the byte size of an entire pixel.
    fn size() -> usize {
        Self::total_channels() * mem::size_of::<Self::Element>()
    }

    /// Returns a new DynPixelFormat representing this PixelFormat.
    fn as_dyn() -> DynPixelFormat {
        DynPixelFormat {
            space: Self::SPACE,
            is_float: Self::IS_FLOAT,
            channels: Self::CHANNELS,
            extra_channels: Self::EXTRA_CHANNELS,
            extra_first: Self::EXTRA_FIRST,
            reverse: Self::REVERSE,
            element_size: mem::size_of::<Self::Element>(),
            user_info: unsafe {
                Arc::new(Box::new(StaticPixelFormatInfo {
                    decode_16_fn: mem::transmute(Self::DECODE_16_FN),
                    decode_float_fn: mem::transmute(Self::DECODE_FLOAT_FN),
                    encode_16_fn: mem::transmute(Self::ENCODE_16_FN),
                    encode_float_fn: mem::transmute(Self::ENCODE_FLOAT_FN),
                }))
            },
            decode_16_fn: decode_16_dyn_proxy,
            decode_float_fn: decode_float_dyn_proxy,
            encode_16_fn: encode_16_dyn_proxy,
            encode_float_fn: encode_float_dyn_proxy,
        }
    }
}

/// A dynamic pixel format that can be created at runtime.
///
/// See PixelFormat for documentation.
#[derive(Clone)]
pub struct DynPixelFormat {
    pub space: ColorSpace,
    pub is_float: bool,
    pub channels: usize,
    pub extra_channels: usize,
    pub extra_first: bool,
    pub reverse: bool,
    pub element_size: usize,

    /// User info for the decode/encode functions.
    pub user_info: Arc<Box<Any + Send + Sync>>,

    pub decode_16_fn: fn(fmt: &DynPixelFormat, data: *const (), buf: &mut [u16; MAX_CHANNELS]),
    pub decode_float_fn: fn(fmt: &DynPixelFormat, data: *const (), buf: &mut [f32; MAX_CHANNELS]),
    pub encode_16_fn: fn(fmt: &DynPixelFormat, data: &[u16; MAX_CHANNELS], buf: *mut ()),
    pub encode_float_fn: fn(fmt: &DynPixelFormat, data: &[f32; MAX_CHANNELS], buf: *mut ()),
}

impl DynPixelFormat {
    pub fn size(&self) -> usize {
        (self.channels + self.extra_channels) * self.element_size
    }
}

fn decode_16_dyn_proxy(fmt: &DynPixelFormat, data: *const (), buf: &mut [u16; MAX_CHANNELS]) {
    let info = fmt
        .user_info
        .downcast_ref::<StaticPixelFormatInfo>()
        .unwrap();
    (info.decode_16_fn)(unsafe { &*(data as *const [u8; MAX_BYTES_PER_PIXEL]) }, buf);
}
fn decode_float_dyn_proxy(fmt: &DynPixelFormat, data: *const (), buf: &mut [f32; MAX_CHANNELS]) {
    let info = fmt
        .user_info
        .downcast_ref::<StaticPixelFormatInfo>()
        .unwrap();
    (info.decode_float_fn)(unsafe { &*(data as *const [u8; MAX_BYTES_PER_PIXEL]) }, buf);
}
fn encode_16_dyn_proxy(fmt: &DynPixelFormat, data: &[u16; MAX_CHANNELS], buf: *mut ()) {
    let info = fmt
        .user_info
        .downcast_ref::<StaticPixelFormatInfo>()
        .unwrap();
    (info.encode_16_fn)(data, unsafe {
        &mut *(buf as *mut [u8; MAX_BYTES_PER_PIXEL])
    });
}
fn encode_float_dyn_proxy(fmt: &DynPixelFormat, data: &[f32; MAX_CHANNELS], buf: *mut ()) {
    let info = fmt
        .user_info
        .downcast_ref::<StaticPixelFormatInfo>()
        .unwrap();
    (info.encode_float_fn)(data, unsafe {
        &mut *(buf as *mut [u8; MAX_BYTES_PER_PIXEL])
    });
}

// implementation

/// One-channel gray.
pub struct Gray<T> {
    _p: PhantomData<T>,
}

/// Simple RGB.
pub struct RGB<T> {
    _p: PhantomData<T>,
}

/// Simple RGB with an alpha channel.
pub struct RGBA<T> {
    _p: PhantomData<T>,
}

/// Simple CMYK.
pub struct CMYK<T> {
    _p: PhantomData<T>,
}

/// L* a* b*.
pub struct Lab<T> {
    _p: PhantomData<T>,
}

macro_rules! decode_def {
    ($t:ty => u16: $name:ident, $($i:expr),+; $fac:expr) => {
        fn $name(src: &[$t], out: &mut [u16; MAX_CHANNELS]) {
            $(
                out[$i] = quick_saturate_word(src[$i] as f64 * $fac);
            )+
        }
    };
    ($t:ty => f32: $name:ident, $($i:expr),+) => {
        fn $name(src: &[$t], out: &mut [f32; MAX_CHANNELS]) {
            $(
                out[$i] = src[$i] as f32;
            )+
        }
    };
}

macro_rules! encode_def {
    (u16 => $t:ty: $name:ident, $($i:expr),+; $fac:expr) => {
        fn $name(src: &[u16; MAX_CHANNELS], out: &mut [$t]) {
            $(
                out[$i] = src[$i] as $t / $fac;
            )+
        }
    };
    (f32 => $t:ty: $name:ident, $($i:expr),+) => {
        fn $name(src: &[f32; MAX_CHANNELS], out: &mut [$t]) {
            $(
                out[$i] = src[$i] as $t;
            )+
        }
    };
}

decode_def!(f32 => u16: decode_16_1_float, 0; 65535.);
decode_def!(f32 => f32: decode_float_1_float, 0);
encode_def!(u16 => f32: encode_16_1_float, 0; 65535.);
encode_def!(f32 => f32: encode_float_1_float, 0);
decode_def!(f64 => u16: decode_16_1_double, 0; 65535.);
decode_def!(f64 => f32: decode_float_1_double, 0);
encode_def!(u16 => f64: encode_16_1_double, 0; 65535.);
encode_def!(f32 => f64: encode_float_1_double, 0);

impl PixelFormat for Gray<f32> {
    type Element = f32;
    const SPACE: ColorSpace = ColorSpace::Gray;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 1;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f32> = decode_16_1_float;
    const DECODE_FLOAT_FN: DecodeFloatFn<f32> = decode_float_1_float;
    const ENCODE_16_FN: Encode16Fn<f32> = encode_16_1_float;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f32> = encode_float_1_float;
}

impl PixelFormat for Gray<f64> {
    type Element = f64;
    const SPACE: ColorSpace = ColorSpace::Gray;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 1;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f64> = decode_16_1_double;
    const DECODE_FLOAT_FN: DecodeFloatFn<f64> = decode_float_1_double;
    const ENCODE_16_FN: Encode16Fn<f64> = encode_16_1_double;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f64> = encode_float_1_double;
}

decode_def!(f32 => u16: decode_16_3_float, 0, 1, 2; 65535.);
decode_def!(f32 => f32: decode_float_3_float, 0, 1, 2);
encode_def!(u16 => f32: encode_16_3_float, 0, 1, 2; 65535.);
encode_def!(f32 => f32: encode_float_3_float, 0, 1, 2);
decode_def!(f64 => u16: decode_16_3_double, 0, 1, 2; 65535.);
decode_def!(f64 => f32: decode_float_3_double, 0, 1, 2);
encode_def!(u16 => f64: encode_16_3_double, 0, 1, 2; 65535.);
encode_def!(f32 => f64: encode_float_3_double, 0, 1, 2);

impl PixelFormat for RGB<f32> {
    type Element = f32;
    const SPACE: ColorSpace = ColorSpace::RGB;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 3;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f32> = decode_16_3_float;
    const DECODE_FLOAT_FN: DecodeFloatFn<f32> = decode_float_3_float;
    const ENCODE_16_FN: Encode16Fn<f32> = encode_16_3_float;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f32> = encode_float_3_float;
}

impl PixelFormat for RGB<f64> {
    type Element = f64;
    const SPACE: ColorSpace = ColorSpace::RGB;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 3;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f64> = decode_16_3_double;
    const DECODE_FLOAT_FN: DecodeFloatFn<f64> = decode_float_3_double;
    const ENCODE_16_FN: Encode16Fn<f64> = encode_16_3_double;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f64> = encode_float_3_double;
}

impl PixelFormat for RGBA<f32> {
    type Element = f32;
    const SPACE: ColorSpace = ColorSpace::RGB;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 3;
    const EXTRA_CHANNELS: usize = 1;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f32> = decode_16_3_float;
    const DECODE_FLOAT_FN: DecodeFloatFn<f32> = decode_float_3_float;
    const ENCODE_16_FN: Encode16Fn<f32> = encode_16_3_float;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f32> = encode_float_3_float;
}

impl PixelFormat for RGBA<f64> {
    type Element = f64;
    const SPACE: ColorSpace = ColorSpace::RGB;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 3;
    const EXTRA_CHANNELS: usize = 1;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f64> = decode_16_3_double;
    const DECODE_FLOAT_FN: DecodeFloatFn<f64> = decode_float_3_double;
    const ENCODE_16_FN: Encode16Fn<f64> = encode_16_3_double;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f64> = encode_float_3_double;
}

decode_def!(f32 => u16: decode_16_4_float, 0, 1, 2, 3; 65535.);
decode_def!(f32 => f32: decode_float_4_float, 0, 1, 2, 3);
encode_def!(u16 => f32: encode_16_4_float, 0, 1, 2, 3; 65535.);
encode_def!(f32 => f32: encode_float_4_float, 0, 1, 2, 3);
decode_def!(f64 => u16: decode_16_4_double, 0, 1, 2, 3; 65535.);
decode_def!(f64 => f32: decode_float_4_double, 0, 1, 2, 3);
encode_def!(u16 => f64: encode_16_4_double, 0, 1, 2, 3; 65535.);
encode_def!(f32 => f64: encode_float_4_double, 0, 1, 2, 3);

impl PixelFormat for CMYK<f32> {
    type Element = f32;
    const SPACE: ColorSpace = ColorSpace::CMYK;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 4;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f32> = decode_16_4_float;
    const DECODE_FLOAT_FN: DecodeFloatFn<f32> = decode_float_4_float;
    const ENCODE_16_FN: Encode16Fn<f32> = encode_16_4_float;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f32> = encode_float_4_float;
}

impl PixelFormat for CMYK<f64> {
    type Element = f64;
    const SPACE: ColorSpace = ColorSpace::CMYK;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 4;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f64> = decode_16_4_double;
    const DECODE_FLOAT_FN: DecodeFloatFn<f64> = decode_float_4_double;
    const ENCODE_16_FN: Encode16Fn<f64> = encode_16_4_double;
    const ENCODE_FLOAT_FN: EncodeFloatFn<f64> = encode_float_4_double;
}

macro_rules! de_en_lab_float_def {
    ($de_name:ident, $en_name:ident, $ty:ty) => {
        fn $de_name(src: &[$ty], out: &mut [f32; 16]) {
            // from 0..100 to 0..1
            out[0] = (src[0] / 100.) as f32;
            // from -128..+127 to 0..1
            out[1] = (src[1] + 128. / 255.) as f32;
            out[2] = (src[2] + 128. / 255.) as f32;
        }

        fn $en_name(src: &[f32; 16], out: &mut [$ty]) {
            out[0] = src[0] as $ty * 100.;
            out[1] = src[1] as $ty * 255. - 128.;
            out[2] = src[2] as $ty * 255. - 128.;
        }
    };
}

de_en_lab_float_def!(decode_float_lab_float, encode_float_lab_float, f32);
de_en_lab_float_def!(decode_float_lab_double, encode_float_lab_double, f64);

impl PixelFormat for Lab<f32> {
    type Element = f32;
    const SPACE: ColorSpace = ColorSpace::Lab;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 3;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f32> = decode_16_3_float; // TODO
    const DECODE_FLOAT_FN: DecodeFloatFn<f32> = decode_float_lab_float;
    const ENCODE_16_FN: Encode16Fn<f32> = encode_16_3_float; // TODO
    const ENCODE_FLOAT_FN: EncodeFloatFn<f32> = encode_float_lab_float;
}

impl PixelFormat for Lab<f64> {
    type Element = f64;
    const SPACE: ColorSpace = ColorSpace::Lab;
    const IS_FLOAT: bool = true;
    const CHANNELS: usize = 3;
    const EXTRA_CHANNELS: usize = 0;
    const EXTRA_FIRST: bool = false;
    const REVERSE: bool = false;
    const DECODE_16_FN: Decode16Fn<f64> = decode_16_3_double; // TODO
    const DECODE_FLOAT_FN: DecodeFloatFn<f64> = decode_float_lab_double;
    const ENCODE_16_FN: Encode16Fn<f64> = encode_16_3_double; // TODO
    const ENCODE_FLOAT_FN: EncodeFloatFn<f64> = encode_float_lab_double;
}

#[doc(hidden)]
pub enum ElType {
    Int, // TODO: actual integer types
    Float,
    Double,
}

impl ElType {
    fn is_float(self) -> bool {
        match self {
            ElType::Float | ElType::Double => true,
            _ => false,
        }
    }
}

#[doc(hidden)]
pub trait PixelFormatElement: Copy {
    fn el_type() -> ElType;
}

macro_rules! impl_pf_element_not_float {
    ($($ty:ty),+) => {
        $(
        impl PixelFormatElement for $ty {
            fn el_type() -> ElType {
                ElType::Int
            }
        }
        )+
    }
}

impl_pf_element_not_float!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);
impl PixelFormatElement for f32 {
    fn el_type() -> ElType {
        ElType::Float
    }
}
impl PixelFormatElement for f64 {
    fn el_type() -> ElType {
        ElType::Double
    }
}

impl ColorSpace {
    /// Creates a pixel format based on the element type and this color space.
    ///
    /// The element type must be one of the number primitives.
    ///
    /// # Examples
    /// ```
    /// # use lcms_prime::ColorSpace;
    /// # use lcms_prime::pixel_format::DynPixelFormat;
    /// let pixel_format: DynPixelFormat = ColorSpace::RGB.pixel_format::<f32>().unwrap();
    /// assert_eq!(pixel_format.space, ColorSpace::RGB);
    /// assert_eq!(pixel_format.is_float, true);
    /// ```
    pub fn pixel_format<Element: PixelFormatElement>(self) -> Result<DynPixelFormat, String> {
        macro_rules! match_cs_el {
            ($(($cs:pat, $el:pat) => $key:ty),+, _ => $else:expr,) => {
                match (self, Element::el_type()) {
                    $(
                    ($cs, $el) => (
                        mem::transmute(<$key as PixelFormat>::DECODE_16_FN),
                        mem::transmute(<$key as PixelFormat>::DECODE_FLOAT_FN),
                        mem::transmute(<$key as PixelFormat>::ENCODE_16_FN),
                        mem::transmute(<$key as PixelFormat>::ENCODE_FLOAT_FN),
                    ),
                    )+
                    _ => $else
                }
            }
        }
        let (decode_16_fn, decode_float_fn, encode_16_fn, encode_float_fn) = unsafe {
            match_cs_el! {
                (ColorSpace::Gray, ElType::Float) => Gray<f32>,
                (ColorSpace::Gray, ElType::Double) => Gray<f64>,
                (ColorSpace::RGB, ElType::Float) => RGB<f32>,
                (ColorSpace::RGB, ElType::Double) => RGB<f64>,
                (ColorSpace::CMYK, ElType::Float) => CMYK<f32>,
                (ColorSpace::CMYK, ElType::Double) => CMYK<f64>,
                (ColorSpace::Lab, ElType::Float) => Lab<f32>,
                (ColorSpace::Lab, ElType::Double) => Lab<f64>,
                _ => return Err("Could not find suitable decoder and encoder functions".into()),
            }
        };

        Ok(DynPixelFormat {
            space: self,
            is_float: Element::el_type().is_float(),
            channels: self.channels(),
            extra_channels: 0,
            extra_first: false,
            reverse: false,
            element_size: mem::size_of::<Element>(),

            user_info: Arc::new(Box::new(StaticPixelFormatInfo {
                decode_16_fn,
                decode_float_fn,
                encode_16_fn,
                encode_float_fn,
            })),

            decode_16_fn: decode_16_dyn_proxy,
            decode_float_fn: decode_float_dyn_proxy,
            encode_16_fn: encode_16_dyn_proxy,
            encode_float_fn: encode_float_dyn_proxy,
        })
    }
}

#[test]
fn dyn_proxies_do_not_break_due_to_excessively_unsafe_code() {
    let rgb = RGB::<f32>::as_dyn();
    let in_buf: &[f32] = &[0.2147, 0.483, 0.647];
    let mut out_buf: [f32; 16] = [0.; 16];
    (rgb.decode_float_fn)(&rgb, in_buf as *const _ as *const (), &mut out_buf);
    assert_eq!(in_buf[0..3], out_buf[0..3]);
}
