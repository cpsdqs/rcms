use internal::quick_saturate_word;
use std::marker::PhantomData;
use std::mem;
use ColorSpace;

pub const MAX_CHANNELS: usize = 16;

/// A function that decodes a single pixel at index 0 of the given array slice and puts the output in the u16 array.
pub type Decode16Fn<T> = fn(&[T], &mut [u16; MAX_CHANNELS]);

/// A function that decodes a single pixel at index 0 of the given array slice and puts the output in the f32 array.
pub type DecodeFloatFn<T> = fn(&[T], &mut [f32; MAX_CHANNELS]);

/// A function that encodes a single pixel to index 0 of the given array slice.
pub type Encode16Fn<T> = fn(&[u16; MAX_CHANNELS], &mut [T]);

/// A function that encodes a single pixel to index 0 of the given array slice.
pub type EncodeFloatFn<T> = fn(&[f32; MAX_CHANNELS], &mut [T]);

pub trait PixelFormat {
    /// The primitive type of the pixel format.
    ///
    /// For example, 8-bit image formats would use u8.
    /// The data will be cast to this type with no further checks.
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

    fn size() -> usize {
        (Self::CHANNELS + Self::EXTRA_CHANNELS) * mem::size_of::<Self::Element>()
    }
}

// implementation

pub struct Gray<T> {
    _p: PhantomData<T>,
}
pub struct RGB<T> {
    _p: PhantomData<T>,
}
pub struct RGBA<T> {
    _p: PhantomData<T>,
}
pub struct CMYK<T> {
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
            println!("{:?}", out);
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
