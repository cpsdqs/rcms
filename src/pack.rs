//! This module handles all formats supported by lcms.
//!
//! There are two flavors: 16 bits and floating point.
//! Floating point is supported only in a subset, those formats holding f32 (4 bytes per component)
//! and double (marked as 0 bytes per component as special case).

use half::{float_to_half, half_to_float};
use internal::quick_saturate_word;
use libc::memmove;
use pcs::{
    float_to_lab_encoded, float_to_xyz_encoded, lab_encoded_to_float, lcms_color_space,
    xyz_encoded_to_float, MAX_ENCODEABLE_XYZ,
};
use profile::Profile;
use std::{fmt, mem, ops};
use transform::Transform;
use {CIELab, PixelType, CIEXYZ};

// This macro return words stored as big endian
macro_rules! change_endian {
    ($w:expr) => {
        ($w << 8) as u16 | ($w >> 8) as u16
    };
}

// These macros handle reversing (negative)
macro_rules! reverse_flavor_8 {
    ($x:expr) => {
        0xFF - ($x as u8)
    };
}
macro_rules! reverse_flavor_16 {
    ($x:expr) => {
        0xFFFF - ($x as u16)
    };
}

/// * 0xffff / 0xff00 = (255 * 257) / (255 * 256) = 257 / 256
fn labv2_to_labv4(x: u16) -> u16 {
    let a = ((x as u32) << 8 | x as u32) >> 8; // * 257 / 256
    if a > 0xFFFF {
        0xFFFF
    } else {
        a as u16
    }
}

/// * 0xf00 / 0xffff = * 256 / 257
fn labv4_to_labv2(x: u16) -> u16 {
    (((x << 8) + 0x80) / 257) as u16
}

/// A 16-bit formatter.
pub type Formatter16 = unsafe fn(&Transform, &mut [u16], Accum, u32) -> Accum;

/// A float formatter.
pub type FormatterFloat = unsafe fn(&Transform, &mut [f32], Accum, u32) -> Accum;

struct Formatters16 {
    ty: u32,
    mask: u32,
    frm: Formatter16,
}

struct FormattersFloat {
    ty: u32,
    mask: u32,
    frm: FormatterFloat,
}

const ANYSPACE: u32 = colorspace_sh!(31);
const ANYCHANNELS: u32 = channels_sh!(15);
const ANYEXTRA: u32 = extra_sh!(7);
const ANYPLANAR: u32 = planar_sh!(1);
const ANYENDIAN: u32 = endian16_sh!(1);
const ANYSWAP: u32 = do_swap_sh!(1);
const ANYSWAPFIRST: u32 = swap_first_sh!(1);
const ANYFLAVOR: u32 = flavor_sh!(1);

/// A pointer type that kind of acts like a cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Accum(*mut u8);

impl Accum {
    pub unsafe fn new(ptr: *mut ()) -> Accum {
        Accum(ptr as *mut _)
    }

    unsafe fn deref_as<T: Copy>(self) -> T {
        *(self.0 as *const T)
    }
    unsafe fn deref_mut<T: Copy>(&mut self) -> &mut T {
        &mut *(self.0 as *mut _)
    }
    fn as_mut(self) -> *mut u8 {
        self.0
    }
    fn offset<T>(self, n: i32) -> Accum {
        self + mem::size_of::<T>() as i32 * n
    }
}

impl ops::Deref for Accum {
    type Target = u8;
    fn deref(&self) -> &u8 {
        unsafe { &*self.0 }
    }
}
impl ops::DerefMut for Accum {
    fn deref_mut(&mut self) -> &mut u8 {
        unsafe { &mut *self.0 }
    }
}
impl ops::Add<u32> for Accum {
    type Output = Accum;
    fn add(self, other: u32) -> Accum {
        Accum(unsafe { self.0.offset(other as isize) })
    }
}
impl ops::Add<usize> for Accum {
    type Output = Accum;
    fn add(self, other: usize) -> Accum {
        Accum(unsafe { self.0.offset(other as isize) })
    }
}
impl ops::Add<i32> for Accum {
    type Output = Accum;
    fn add(self, other: i32) -> Accum {
        Accum(unsafe { self.0.offset(other as isize) })
    }
}
impl ops::AddAssign<i32> for Accum {
    fn add_assign(&mut self, other: i32) {
        self.0 = unsafe { self.0.offset(other as isize) };
    }
}
impl ops::AddAssign<u32> for Accum {
    fn add_assign(&mut self, other: u32) {
        self.0 = unsafe { self.0.offset(other as isize) };
    }
}
impl ops::AddAssign<usize> for Accum {
    fn add_assign(&mut self, other: usize) {
        self.0 = unsafe { self.0.offset(other as isize) };
    }
}

// unpacking routines (16 bits)

/// Does almost everything but is slow
unsafe fn unroll_chunky_bytes(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;

    if extra_first {
        accum += extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        let mut v = from_8_to_16!(*accum);
        v = if reverse { reverse_flavor_16!(v) } else { v };
        w_in[index as usize] = v;
        accum += 1;
    }

    if !extra_first {
        accum += extra;
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    accum
}

/// Extra channels are just ignored because they come in the next planes
unsafe fn unroll_planar_bytes(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let init = accum;

    if do_swap ^ swap_first {
        accum += t_extra!(info.input_format) * stride;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        let v = from_8_to_16!(*accum);

        w_in[index as usize] = if reverse { reverse_flavor_16!(v) } else { v };
        accum += stride;
    }

    init + 1
}

// Special cases, provided for performance
unsafe fn unroll_4_bytes(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // C
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;
    // M
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // Y
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;
    // K
    w_in[3] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

unsafe fn unroll_4_bytes_reverse(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // C
    w_in[0] = from_8_to_16!(reverse_flavor_8!(*accum));
    accum += 1;
    // M
    w_in[1] = from_8_to_16!(reverse_flavor_8!(*accum));
    accum += 1;
    // Y
    w_in[2] = from_8_to_16!(reverse_flavor_8!(*accum));
    accum += 1;
    // K
    w_in[3] = from_8_to_16!(reverse_flavor_8!(*accum));
    accum += 1;

    accum
}

unsafe fn unroll_4_bytes_swap_first(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // K
    w_in[3] = from_8_to_16!(*accum);
    accum += 1;
    // C
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;
    // M
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // Y
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

/// KYMC
unsafe fn unroll_4_bytes_swap(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // K
    w_in[3] = from_8_to_16!(*accum);
    accum += 1;
    // Y
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;
    // M
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // C
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

unsafe fn unroll_4_bytes_swap_swap_first(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // K
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;
    // Y
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // M
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;
    // C
    w_in[3] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

unsafe fn unroll_3_bytes(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // R
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;
    // G
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // B
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

unsafe fn unroll_3_bytes_skip_1_swap(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    accum += 1; // A
                // B
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;
    // G
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // R
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

unsafe fn unroll_3_bytes_skip_1_swap_swap_first(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // B
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;
    // G
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // R
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;
    accum += 1; // A

    accum
}

unsafe fn unroll_3_bytes_skip_1_swap_first(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    accum += 1; // A
                // R
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;
    // G
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // B
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

/// BRG
unsafe fn unroll_3_bytes_swap(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // B
    w_in[2] = from_8_to_16!(*accum);
    accum += 1;
    // G
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;
    // R
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

unsafe fn unroll_labv2_8(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = labv2_to_labv4(from_8_to_16!(*accum));
    accum += 1;
    // a
    w_in[1] = labv2_to_labv4(from_8_to_16!(*accum));
    accum += 1;
    // b
    w_in[2] = labv2_to_labv4(from_8_to_16!(*accum));
    accum += 1;

    accum
}

unsafe fn unroll_a_labv2_8(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    accum += 1; // A
                // L
    w_in[0] = labv2_to_labv4(from_8_to_16!(*accum));
    accum += 1;
    // a
    w_in[1] = labv2_to_labv4(from_8_to_16!(*accum));
    accum += 1;
    // b
    w_in[2] = labv2_to_labv4(from_8_to_16!(*accum));
    accum += 1;

    accum
}

unsafe fn unroll_labv2_16(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = labv2_to_labv4(accum.deref_as::<u16>());
    accum += 2;
    // a
    w_in[1] = labv2_to_labv4(accum.deref_as::<u16>());
    accum += 2;
    // b
    w_in[2] = labv2_to_labv4(accum.deref_as::<u16>());
    accum += 2;

    accum
}

/// for duplex
unsafe fn unroll_2_bytes(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // ch1
    w_in[0] = from_8_to_16!(*accum);
    accum += 1;
    // ch2
    w_in[1] = from_8_to_16!(*accum);
    accum += 1;

    accum
}

/// Monochrome duplicates L into RGB for null-transforms
unsafe fn unroll_1_byte(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = from_8_to_16!(*accum);
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];
    accum += 1;

    accum
}

unsafe fn unroll_1_byte_skip_1(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = from_8_to_16!(*accum);
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];
    accum += 1;
    accum += 1;

    accum
}

unsafe fn unroll_1_byte_skip_2(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = from_8_to_16!(*accum);
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];
    accum += 1;
    accum += 2;

    accum
}

unsafe fn unroll_1_byte_reversed(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = reverse_flavor_16!(from_8_to_16!(*accum));
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];
    accum += 1;

    accum
}

unsafe fn unroll_any_words(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let swap_endian = t_endian16!(info.input_format) != 0;
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;

    if extra_first {
        accum += extra * mem::size_of::<u16>() as u32;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };
        let mut v = accum.deref_as::<u16>();

        if swap_endian {
            v = change_endian!(v);
        }

        w_in[index as usize] = if reverse { reverse_flavor_16!(v) } else { v };

        accum += mem::size_of::<u16>();
    }

    if !extra_first {
        accum += extra * mem::size_of::<u16>() as u32;
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    accum
}

unsafe fn unroll_planar_words(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let swap_endian = t_endian16!(info.input_format) != 0;
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let init = accum;

    if do_swap {
        accum += t_extra!(info.input_format) * stride;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };
        let mut v = accum.deref_as::<u16>();

        if swap_endian {
            v = change_endian!(v);
        }

        w_in[index as usize] = if reverse { reverse_flavor_16!(v) } else { v };

        accum += stride;
    }

    init + mem::size_of::<u16>()
}

unsafe fn unroll_4_words(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // C
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;
    // M
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // Y
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;
    // K
    w_in[3] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

unsafe fn unroll_4_words_reverse(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // C
    w_in[0] = reverse_flavor_16!(accum.deref_as::<u16>());
    accum += 2;
    // M
    w_in[1] = reverse_flavor_16!(accum.deref_as::<u16>());
    accum += 2;
    // Y
    w_in[2] = reverse_flavor_16!(accum.deref_as::<u16>());
    accum += 2;
    // K
    w_in[3] = reverse_flavor_16!(accum.deref_as::<u16>());
    accum += 2;

    accum
}

unsafe fn unroll_4_words_swap_first(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // K
    w_in[3] = accum.deref_as::<u16>();
    accum += 2;
    // C
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;
    // M
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // Y
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

/// KYMC
unsafe fn unroll_4_words_swap(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // K
    w_in[3] = accum.deref_as::<u16>();
    accum += 2;
    // Y
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;
    // M
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // C
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

unsafe fn unroll_4_words_swap_swap_first(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // K
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;
    // Y
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // M
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;
    // C
    w_in[3] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

unsafe fn unroll_3_words(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // C R
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;
    // M G
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // Y B
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

unsafe fn unroll_3_words_swap(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // C R
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;
    // M G
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // Y B
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

unsafe fn unroll_3_words_skip_1_swap(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    accum += 2; // A
                // R
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;
    // G
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // B
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

unsafe fn unroll_3_words_skip_1_swap_first(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    accum += 2; // A
                // R
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;
    // G
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;
    // B
    w_in[2] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

unsafe fn unroll_1_word(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = accum.deref_as::<u16>();
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];
    accum += 2;

    accum
}

unsafe fn unroll_1_word_reversed(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // L
    w_in[0] = reverse_flavor_16!(accum.deref_as::<u16>());
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];
    accum += 2;

    accum
}

unsafe fn unroll_1_word_skip_3(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    w_in[0] = accum.deref_as::<u16>();
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];

    accum += 8;

    accum
}

unsafe fn unroll_2_words(
    _: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    _: u32,
) -> Accum {
    // ch1
    w_in[0] = accum.deref_as::<u16>();
    accum += 2;
    // ch2
    w_in[1] = accum.deref_as::<u16>();
    accum += 2;

    accum
}

/// This is a conversion of Lab doubles to 16 bits
unsafe fn unroll_lab_double_to_16(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let lab = CIELab {
            L: accum.deref_as::<f64>(),
            a: (accum + stride).deref_as::<f64>(),
            b: (accum + stride * 2).deref_as::<f64>(),
        };

        w_in[0..3].copy_from_slice(&float_to_lab_encoded(lab));
        accum + mem::size_of::<f64>()
    } else {
        w_in[0..3].copy_from_slice(&float_to_lab_encoded(accum.deref_as::<CIELab>()));
        accum +=
            mem::size_of::<CIELab>() + t_extra!(info.input_format) as usize * mem::size_of::<f64>();
        accum
    }
}

/// This is a conversion of Lab floats to 16 bits
unsafe fn unroll_lab_float_to_16(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let lab = CIELab {
            L: accum.deref_as::<f32>() as f64,
            a: (accum + stride).deref_as::<f32>() as f64,
            b: (accum + stride * 2).deref_as::<f32>() as f64,
        };

        w_in[0..3].copy_from_slice(&float_to_lab_encoded(lab));
        accum + mem::size_of::<f32>()
    } else {
        let lab = CIELab {
            L: accum.deref_as::<f32>() as f64,
            a: accum.offset::<f32>(1).deref_as::<f32>() as f64,
            b: accum.offset::<f32>(2).deref_as::<f32>() as f64,
        };

        w_in[0..3].copy_from_slice(&float_to_lab_encoded(lab));
        accum += (3 + t_extra!(info.input_format)) as usize * mem::size_of::<f32>();
        accum
    }
}

/// This is a conversion of XYZ doubles to 16 bits
unsafe fn unroll_xyz_double_to_16(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let xyz = CIEXYZ {
            x: accum.deref_as::<f64>(),
            y: (accum + stride).deref_as::<f64>(),
            z: (accum + stride * 2).deref_as::<f64>(),
        };

        w_in[0..3].copy_from_slice(&float_to_xyz_encoded(xyz));
        accum + mem::size_of::<f64>()
    } else {
        w_in[0..3].copy_from_slice(&float_to_xyz_encoded(accum.deref_as::<CIEXYZ>()));
        accum +=
            mem::size_of::<CIEXYZ>() + t_extra!(info.input_format) as usize * mem::size_of::<f64>();

        accum
    }
}

// This is a conversion of XYZ floats to 16 bits
unsafe fn unroll_xyz_float_to_16(
    info: &Transform,
    w_in: &mut [u16],
    mut accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let xyz = CIEXYZ {
            x: accum.deref_as::<f32>() as f64,
            y: (accum + stride).deref_as::<f32>() as f64,
            z: (accum + stride * 2).deref_as::<f32>() as f64,
        };

        w_in[0..3].copy_from_slice(&float_to_xyz_encoded(xyz));
        accum + mem::size_of::<f32>()
    } else {
        let xyz = CIEXYZ {
            x: accum.deref_as::<f32>() as f64,
            y: accum.offset::<f32>(1).deref_as::<f32>() as f64,
            z: accum.offset::<f32>(2).deref_as::<f32>() as f64,
        };

        w_in[0..3].copy_from_slice(&float_to_xyz_encoded(xyz));
        accum += 3 * mem::size_of::<f32>()
            + t_extra!(info.input_format) as usize * mem::size_of::<f32>();

        accum
    }
}

/// Check if space is marked as ink
fn is_ink_space(p_type: u32) -> bool {
    match t_colorspace!(p_type) {
        k if k == PixelType::CMY as u32
        || k == PixelType::CMYK as u32
        || k == PixelType::MCH5 as u32
        || k == PixelType::MCH6 as u32
        || k == PixelType::MCH7 as u32
        || k == PixelType::MCH8 as u32
        || k == PixelType::MCH9 as u32
        || k == PixelType::MCH10 as u32
        || k == PixelType::MCH11 as u32
        || k == PixelType::MCH12 as u32
        || k == PixelType::MCH13 as u32
        || k == PixelType::MCH14 as u32
        || k == PixelType::MCH15 as u32 => true,
        _ => false,
    }
}

/// Return the size in bytes of a given formatter
fn pixel_size(format: u32) -> u32 {
    let fmt_bytes = t_bytes!(format);

    if fmt_bytes == 0 {
        // For double, the T_BYTES field is zero
        mem::size_of::<f64>() as u32
    } else {
        // Otherwise, it is already correct for all formats
        fmt_bytes
    }
}

/// Inks come in percentages, remaining cases are between 0..1.0, again to 16 bits
unsafe fn unroll_double_to_16(
    info: &Transform,
    w_in: &mut [u16],
    accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.input_format) != 0;

    let mut v: f64;
    let mut vi: u16;
    let mut start = 0;
    let maximum = if is_ink_space(info.input_format) {
        655.35
    } else {
        65535.0
    };

    let stride = stride / pixel_size(info.input_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        if planar {
            v = (accum + (i + start) * stride).deref_as::<f64>();
        } else {
            v = (accum + i + start).deref_as::<f64>();
        }

        vi = quick_saturate_word(v * maximum);

        if reverse {
            vi = reverse_flavor_16!(vi);
        }

        w_in[index as usize] = vi;
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    if planar {
        accum + mem::size_of::<f64>()
    } else {
        accum + (n_chan + extra) * mem::size_of::<f64>() as u32
    }
}

unsafe fn unroll_float_to_16(
    info: &Transform,
    w_in: &mut [u16],
    accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.input_format) != 0;

    let mut v: f32;
    let mut vi: u16;
    let mut start = 0;
    let maximum = if is_ink_space(info.input_format) {
        655.35
    } else {
        65535.0
    };

    let stride = stride / pixel_size(info.input_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        if planar {
            v = (accum + (i + start) * stride).deref_as::<f32>();
        } else {
            v = (accum + i + start).deref_as::<f32>();
        }

        vi = quick_saturate_word((v * maximum).into());

        if reverse {
            vi = reverse_flavor_16!(vi);
        }

        w_in[index as usize] = vi;
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    if planar {
        accum + mem::size_of::<f32>()
    } else {
        accum + (n_chan + extra) * mem::size_of::<f32>() as u32
    }
}

/// For 1 channel, we need to duplicate data (it comes in 0..1.0 range)
unsafe fn unroll_double_1_chan(
    _: &Transform,
    w_in: &mut [u16],
    accum: Accum,
    _: u32,
) -> Accum {
    let inks = accum.deref_as::<f64>();

    w_in[0] = quick_saturate_word(inks * 65535.);
    w_in[1] = w_in[0];
    w_in[2] = w_in[0];

    accum + mem::size_of::<f64>()
}

/// For anything going from cmsFloat32Number
unsafe fn unroll_floats_to_float(
    info: &Transform,
    w_in: &mut [f32],
    accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.input_format) != 0;

    let mut v: f32;
    let mut start = 0;
    let maximum = if is_ink_space(info.input_format) {
        100.
    } else {
        1.
    };

    let stride = stride / pixel_size(info.input_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        if planar {
            v = accum.offset::<f32>(((i + start) * stride) as i32).deref_as::<f32>();
        } else {
            v = accum.offset::<f32>((i + start) as i32).deref_as::<f32>();
        }

        v /= maximum;

        w_in[index as usize] = if reverse { 1. - v } else { v };
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<f32>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    if planar {
        accum + mem::size_of::<f32>()
    } else {
        accum + (n_chan + extra) * mem::size_of::<f32>() as u32
    }
}

/// For anything going from double
unsafe fn unroll_doubles_to_float(
    info: &Transform,
    w_in: &mut [f32],
    accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.input_format) != 0;

    let mut v: f64;
    let mut start = 0;
    let maximum = if is_ink_space(info.input_format) {
        100.
    } else {
        1.
    };

    let stride = stride / pixel_size(info.input_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        if planar {
            v = (accum + (i + start) * stride).deref_as::<f64>();
        } else {
            v = (accum + i + start).deref_as::<f64>();
        }

        v /= maximum;

        w_in[index as usize] = if reverse { 1. - v } else { v } as f32;
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<f32>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    if planar {
        accum + mem::size_of::<f64>()
    } else {
        accum + (n_chan + extra) * mem::size_of::<f64>() as u32
    }
}

/// From Lab double to cmsFloat32Number
unsafe fn unroll_lab_double_to_float(
    info: &Transform,
    w_in: &mut [f32],
    accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let stride = stride / pixel_size(info.input_format);

        // from 0..100 to 0..1
        w_in[0] = (accum.deref_as::<f64>() / 100.) as f32;
        // from -128..+127 to 0..1
        w_in[1] =
            (accum.offset::<f64>(stride as i32).deref_as::<f64>() + 128. / 255.) as f32;
        w_in[2] = (accum.offset::<f64>(stride as i32 * 2).deref_as::<f64>()
            + 128. / 255.) as f32;

        accum + mem::size_of::<f64>()
    } else {
        // from 0..100 to 0..1
        w_in[0] = (accum.deref_as::<f64>() / 100.) as f32;
        // from -128..+127 to 0..1
        w_in[1] = (accum.offset::<f64>(1).deref_as::<f64>() + 128. / 255.) as f32;
        w_in[2] = (accum.offset::<f64>(2).deref_as::<f64>() + 128. / 255.) as f32;

        accum + mem::size_of::<f64>() as u32 * (3 + t_extra!(info.input_format))
    }
}

/// From Lab double to cmsFloat32Number
unsafe fn unroll_lab_float_to_float(
    info: &Transform,
    w_in: &mut [f32],
    accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let stride = stride / pixel_size(info.input_format);

        // from 0..100 to 0..1
        w_in[0] = accum.deref_as::<f32>() / 100.;
        // from -128..+127 to 0..1
        w_in[1] = accum.offset::<f32>(stride as i32).deref_as::<f32>() + 128. / 255.;
        w_in[2] = accum.offset::<f32>(stride as i32 * 2).deref_as::<f32>() + 128. / 255.;

        accum + mem::size_of::<f32>()
    } else {
        // from 0..100 to 0..1
        w_in[0] = accum.deref_as::<f32>() / 100.;
        // from -128..+127 to 0..1
        w_in[1] = accum.offset::<f32>(1).deref_as::<f32>() + 128. / 255.;
        w_in[2] = accum.offset::<f32>(2).deref_as::<f32>() + 128. / 255.;

        accum + mem::size_of::<f32>() as u32 * (3 + t_extra!(info.input_format))
    }
}

// 1.15 fixed point, that means maximum value is MAX_ENCODEABLE_XYZ (0xFFFF)
unsafe fn unroll_xyz_double_to_float(
    info: &Transform,
    w_in: &mut [f32],
    accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let stride = stride / pixel_size(info.input_format);

        w_in[0] = (accum.deref_as::<f64>() / MAX_ENCODEABLE_XYZ) as f32;
        w_in[1] = (accum.offset::<f64>(stride as i32).deref_as::<f64>()
            / MAX_ENCODEABLE_XYZ) as f32;
        w_in[2] = (accum.offset::<f64>(stride as i32 * 2).deref_as::<f64>()
            / MAX_ENCODEABLE_XYZ) as f32;

        accum + mem::size_of::<f64>()
    } else {
        w_in[0] = (accum.deref_as::<f64>() / MAX_ENCODEABLE_XYZ) as f32;
        w_in[1] = (accum.offset::<f64>(1).deref_as::<f64>() / MAX_ENCODEABLE_XYZ) as f32;
        w_in[2] = (accum.offset::<f64>(2).deref_as::<f64>() / MAX_ENCODEABLE_XYZ) as f32;

        accum + mem::size_of::<f64>() as u32 * (3 + t_extra!(info.input_format))
    }
}

unsafe fn unroll_xyz_float_to_float(
    info: &Transform,
    w_in: &mut [f32],
    accum: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.input_format) != 0 {
        let stride = stride / pixel_size(info.input_format);

        w_in[0] = accum.deref_as::<f32>() / MAX_ENCODEABLE_XYZ as f32;
        w_in[1] = accum.offset::<f32>(stride as i32).deref_as::<f32>()
            / MAX_ENCODEABLE_XYZ as f32;
        w_in[2] = accum.offset::<f32>(stride as i32 * 2).deref_as::<f32>()
            / MAX_ENCODEABLE_XYZ as f32;

        accum + mem::size_of::<f32>()
    } else {
        w_in[0] = accum.deref_as::<f32>() / MAX_ENCODEABLE_XYZ as f32;
        w_in[1] = accum.offset::<f32>(1).deref_as::<f32>() / MAX_ENCODEABLE_XYZ as f32;
        w_in[2] = accum.offset::<f32>(2).deref_as::<f32>() / MAX_ENCODEABLE_XYZ as f32;

        accum + mem::size_of::<f32>() as u32 * (3 + t_extra!(info.input_format))
    }
}

// Generic chunky for byte
unsafe fn pack_any_bytes(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;

    let mut swap1 = output;
    let mut v: u8 = 0;

    if extra_first {
        output += extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        v = from_16_to_8!(w_out[index as usize]);

        if reverse {
            v = reverse_flavor_8!(v);
        }

        *output = v;
        output += 1;
    }

    if !extra_first {
        output += extra;
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            n_chan as usize - 1,
        );
        *swap1 = v;
    }

    output
}

unsafe fn pack_any_words(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let swap_endian = t_endian16!(info.output_format) != 0;
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;

    let mut v: u16 = 0;
    let mut swap1 = output;

    if extra_first {
        output += extra * mem::size_of::<u16>() as u32;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        v = w_out[index as usize];

        if swap_endian {
            v = change_endian!(v);
        }

        if reverse {
            v = reverse_flavor_16!(v);
        }

        *output.deref_mut() = v;
        output += mem::size_of::<u16>()
    }

    if !extra_first {
        output += extra * mem::size_of::<u16>() as u32;
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        *swap1.deref_mut() = v;
    }

    output
}

unsafe fn pack_planar_bytes(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;

    let init = output;

    if do_swap ^ swap_first {
        output += t_extra!(info.output_format) * stride;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };
        let v = from_16_to_8!(w_out[index as usize]);

        *output = if reverse { reverse_flavor_8!(v) } else { v } as u8;
        output += stride;
    }

    init + 1
}

unsafe fn pack_planar_words(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_endian = t_endian16!(info.output_format) != 0;

    let init = output;
    let mut v: u16;

    if do_swap {
        output += t_extra!(info.output_format) * stride;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        v = w_out[index as usize];

        if swap_endian {
            v = change_endian!(v);
        }

        if reverse {
            v = reverse_flavor_16!(v);
        }

        *output.deref_mut() = v;
        output += stride;
    }

    init + mem::size_of::<u16>()
}

/// CMYKcm (unrolled for speed)
unsafe fn pack_6_bytes(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[3]);
    output += 1;
    *output = from_16_to_8!(w_out[4]);
    output += 1;
    *output = from_16_to_8!(w_out[5]);
    output += 1;

    output
}

/// KCMYcm
unsafe fn pack_6_bytes_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[5]);
    output += 1;
    *output = from_16_to_8!(w_out[4]);
    output += 1;
    *output = from_16_to_8!(w_out[3]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;

    output
}

/// CMYKcm
unsafe fn pack_6_words(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[0];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;
    *output.deref_mut() = w_out[3];
    output += 2;
    *output.deref_mut() = w_out[4];
    output += 2;
    *output.deref_mut() = w_out[5];
    output += 2;

    output
}

/// KCMYcm
unsafe fn pack_6_words_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[5];
    output += 2;
    *output.deref_mut() = w_out[4];
    output += 2;
    *output.deref_mut() = w_out[3];
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[0];
    output += 2;

    output
}

unsafe fn pack_4_bytes(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[3]);
    output += 1;

    output
}

/// KCMYcm
unsafe fn pack_4_bytes_reverse(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = reverse_flavor_8!(from_16_to_8!(w_out[5]));
    output += 1;
    *output = reverse_flavor_8!(from_16_to_8!(w_out[4]));
    output += 1;
    *output = reverse_flavor_8!(from_16_to_8!(w_out[3]));
    output += 1;
    *output = reverse_flavor_8!(from_16_to_8!(w_out[2]));
    output += 1;

    output
}

unsafe fn pack_4_bytes_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[3]);
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;

    output
}

/// ABGR
unsafe fn pack_4_bytes_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[3]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;

    output
}

unsafe fn pack_4_bytes_swap_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    *output = from_16_to_8!(w_out[3]);
    output += 1;

    output
}

unsafe fn pack_4_words(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[0];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;
    *output.deref_mut() = w_out[3];
    output += 2;

    output
}

unsafe fn pack_4_words_reverse(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = reverse_flavor_16!(w_out[0]);
    output += 2;
    *output.deref_mut() = reverse_flavor_16!(w_out[1]);
    output += 2;
    *output.deref_mut() = reverse_flavor_16!(w_out[2]);
    output += 2;
    *output.deref_mut() = reverse_flavor_16!(w_out[3]);
    output += 2;

    output
}

/// ABGR
unsafe fn pack_4_words_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[3];
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[0];
    output += 2;

    output
}

/// CMYK
unsafe fn pack_4_words_big_endian(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = change_endian!(w_out[0]);
    output += 2;
    *output.deref_mut() = change_endian!(w_out[1]);
    output += 2;
    *output.deref_mut() = change_endian!(w_out[2]);
    output += 2;
    *output.deref_mut() = change_endian!(w_out[3]);
    output += 2;

    output
}

unsafe fn pack_labv2_8(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(labv4_to_labv2(w_out[0]));
    output += 1;
    *output = from_16_to_8!(labv4_to_labv2(w_out[1]));
    output += 1;
    *output = from_16_to_8!(labv4_to_labv2(w_out[2]));
    output += 1;

    output
}

unsafe fn pack_a_labv2_8(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 1;
    *output = from_16_to_8!(labv4_to_labv2(w_out[0]));
    output += 1;
    *output = from_16_to_8!(labv4_to_labv2(w_out[1]));
    output += 1;
    *output = from_16_to_8!(labv4_to_labv2(w_out[2]));
    output += 1;

    output
}

unsafe fn pack_labv2_16(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = labv4_to_labv2(w_out[0]);
    output += 2;
    *output.deref_mut() = labv4_to_labv2(w_out[1]);
    output += 2;
    *output.deref_mut() = labv4_to_labv2(w_out[2]);
    output += 2;

    output
}

unsafe fn pack_3_bytes(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;

    output
}

/// (note: I’m not sure if this still has any advantage in the Rust version)
unsafe fn pack_3_bytes_optimized(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = (w_out[0] & 0xFF) as u8;
    output += 1;
    *output = (w_out[1] & 0xFF) as u8;
    output += 1;
    *output = (w_out[2] & 0xFF) as u8;
    output += 1;

    output
}

unsafe fn pack_3_bytes_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;

    output
}

/// (note: I’m not sure if this still has any advantage in the Rust version)
unsafe fn pack_3_bytes_swap_optimized(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = (w_out[2] & 0xFF) as u8;
    output += 1;
    *output = (w_out[1] & 0xFF) as u8;
    output += 1;
    *output = (w_out[0] & 0xFF) as u8;
    output += 1;

    output
}

unsafe fn pack_3_words(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[0];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;

    output
}

unsafe fn pack_3_words_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[2];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[0];
    output += 2;

    output
}

unsafe fn pack_3_words_big_endian(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = change_endian!(w_out[0]);
    output += 2;
    *output.deref_mut() = change_endian!(w_out[1]);
    output += 2;
    *output.deref_mut() = change_endian!(w_out[2]);
    output += 2;

    output
}

unsafe fn pack_3_bytes_and_skip_1(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    output += 1;

    output
}

/// (note: I’m not sure if this still has any advantage in the Rust version)
unsafe fn pack_3_bytes_and_skip_1_optimized(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = (w_out[0] & 0xFF) as u8;
    output += 1;
    *output = (w_out[1] & 0xFF) as u8;
    output += 1;
    *output = (w_out[2] & 0xFF) as u8;
    output += 1;
    output += 1;

    output
}

unsafe fn pack_3_bytes_and_skip_1_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;

    output
}

/// (note: I’m not sure if this still has any advantage in the Rust version)
unsafe fn pack_3_bytes_and_skip_1_swap_first_optimized(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 1;
    *output = (w_out[0] & 0xFF) as u8;
    output += 1;
    *output = (w_out[1] & 0xFF) as u8;
    output += 1;
    *output = (w_out[2] & 0xFF) as u8;
    output += 1;

    output
}

unsafe fn pack_3_bytes_and_skip_1_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 1;
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;

    output
}

/// (note: I’m not sure if this still has any advantage in the Rust version)
unsafe fn pack_3_bytes_and_skip_1_swap_optimized(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 1;
    *output = (w_out[2] & 0xFF) as u8;
    output += 1;
    *output = (w_out[1] & 0xFF) as u8;
    output += 1;
    *output = (w_out[0] & 0xFF) as u8;
    output += 1;

    output
}

unsafe fn pack_3_bytes_and_skip_1_swap_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[2]);
    output += 1;
    *output = from_16_to_8!(w_out[1]);
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;
    output += 1;

    output
}

/// (note: I’m not sure if this still has any advantage in the Rust version)
unsafe fn pack_3_bytes_and_skip_1_swap_swap_first_optimized(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = (w_out[2] & 0xFF) as u8;
    output += 1;
    *output = (w_out[1] & 0xFF) as u8;
    output += 1;
    *output = (w_out[0] & 0xFF) as u8;
    output += 1;
    output += 1;

    output
}

unsafe fn pack_3_words_and_skip_1(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[0];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;
    output += 2;

    output
}

unsafe fn pack_3_words_and_skip_1_swap(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[0];
    output += 2;

    output
}

unsafe fn pack_3_words_and_skip_1_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 2;
    *output.deref_mut() = w_out[0];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[2];
    output += 2;

    output
}

unsafe fn pack_3_words_and_skip_1_swap_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[2];
    output += 2;
    *output.deref_mut() = w_out[1];
    output += 2;
    *output.deref_mut() = w_out[0];
    output += 2;
    output += 2;

    output
}

unsafe fn pack_1_byte(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[0]);
    output += 1;

    output
}

unsafe fn pack_1_byte_reversed(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(reverse_flavor_16!(w_out[0]));
    output += 1;

    output
}

unsafe fn pack_1_byte_skip_1(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output = from_16_to_8!(w_out[0]);
    output += 2;

    output
}

unsafe fn pack_1_byte_skip_1_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 1;
    *output = from_16_to_8!(w_out[0]);
    output += 1;

    output
}

unsafe fn pack_1_word(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[0];
    output += 2;

    output
}

unsafe fn pack_1_word_reversed(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = reverse_flavor_16!(w_out[0]);
    output += 2;

    output
}

unsafe fn pack_1_word_big_endian(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = change_endian!(w_out[0]);
    output += 2;

    output
}

unsafe fn pack_1_word_skip_1(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    *output.deref_mut() = w_out[0];
    output += 4;

    output
}

unsafe fn pack_1_word_skip_1_swap_first(
    _: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    _: u32,
) -> Accum {
    output += 2;
    *output.deref_mut() = w_out[0];
    output += 2;

    output
}

/// Unencoded Float values—don't try optimize speed
unsafe fn pack_lab_double_from_16(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.output_format) != 0 {
        let lab = lab_encoded_to_float(&w_out[0..3]);
        // TODO: see if a stride / pixel_size is missing here (there’s none in the original?)
        *output.deref_mut::<f64>() = lab.L;
        *output.offset::<f64>(stride as i32).deref_mut::<f64>() = lab.a;
        *output.offset::<f64>(stride as i32 * 2).deref_mut::<f64>() = lab.b;

        output + mem::size_of::<f64>()
    } else {
        *output.deref_mut::<CIELab>() = lab_encoded_to_float(&w_out[0..3]);
        output
            + mem::size_of::<CIELab>()
            + t_extra!(info.output_format) as usize * mem::size_of::<f64>()
    }
}

unsafe fn pack_lab_float_from_16(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    let lab = lab_encoded_to_float(&w_out[0..3]);
    if t_planar!(info.output_format) != 0 {
        let stride = stride / pixel_size(info.output_format);
        *output.deref_mut::<f32>() = lab.L as f32;
        *output.offset::<f32>(stride as i32).deref_mut::<f32>() = lab.a as f32;
        *output.offset::<f32>(stride as i32 * 2).deref_mut::<f32>() = lab.b as f32;
        output + mem::size_of::<f32>()
    } else {
        *output.deref_mut::<f32>() = lab.L as f32;
        *output.offset::<f32>(1).deref_mut::<f32>() = lab.a as f32;
        *output.offset::<f32>(2).deref_mut::<f32>() = lab.b as f32;
        output + (3 + t_extra!(info.output_format)) + mem::size_of::<f32>()
    }
}

unsafe fn pack_xyz_double_from_16(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.output_format) != 0 {
        let xyz = xyz_encoded_to_float(&w_out[0..3]);
        let stride = stride / pixel_size(info.output_format);
        *output.deref_mut::<f64>() = xyz.x;
        *output.offset::<f64>(stride as i32).deref_mut::<f64>() = xyz.y;
        *output.offset::<f64>(stride as i32 * 2).deref_mut::<f64>() = xyz.z;

        output + mem::size_of::<f64>()
    } else {
        *output.deref_mut::<CIEXYZ>() = xyz_encoded_to_float(&w_out[0..3]);
        output
            + mem::size_of::<CIEXYZ>()
            + t_extra!(info.output_format) as usize * mem::size_of::<f64>()
    }
}

unsafe fn pack_xyz_float_from_16(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    let xyz = xyz_encoded_to_float(&w_out[0..3]);
    if t_planar!(info.output_format) != 0 {
        let stride = stride / pixel_size(info.output_format);
        *output.deref_mut::<f32>() = xyz.x as f32;
        *output.offset::<f32>(stride as i32).deref_mut::<f32>() = xyz.y as f32;
        *output.offset::<f32>(stride as i32 * 2).deref_mut::<f32>() = xyz.z as f32;
        output + mem::size_of::<f32>()
    } else {
        *output.deref_mut::<f32>() = xyz.x as f32;
        *output.offset::<f32>(1).deref_mut::<f32>() = xyz.y as f32;
        *output.offset::<f32>(2).deref_mut::<f32>() = xyz.z as f32;
        output + (3 + t_extra!(info.output_format)) + mem::size_of::<f32>()
    }
}

unsafe fn pack_double_from_16(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.output_format) != 0;

    let maximum = if is_ink_space(info.output_format) {
        655.35
    } else {
        65535.
    };
    let mut v: f64 = 0.;
    let mut swap1 = output;
    let mut start = 0;

    let stride = stride / pixel_size(info.output_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        v = w_out[index as usize] as f64 / maximum;

        if reverse {
            v = maximum - v;
        }

        if planar {
            *output
                .offset::<f64>(((i + start) * stride) as i32)
                .deref_mut() = v;
        } else {
            *output.offset::<f64>((i + start) as i32).deref_mut() = v;
        }
    }

    if !extra_first {
        output += extra * mem::size_of::<u16>() as u32;
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            (n_chan - 1) as usize * mem::size_of::<f64>(),
        );
        *swap1.deref_mut() = v;
    }

    if planar {
        output + mem::size_of::<f64>()
    } else {
        output + (n_chan + extra) * mem::size_of::<f64>() as u32
    }
}

unsafe fn pack_float_from_16(
    info: &Transform,
    w_out: &mut [u16],
    mut output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.output_format) != 0;

    let maximum = if is_ink_space(info.output_format) {
        655.35
    } else {
        65535.
    };
    let mut v: f64 = 0.;
    let mut swap1 = output;
    let mut start = 0;

    let stride = stride / pixel_size(info.output_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        v = w_out[index as usize] as f64 / maximum;

        if reverse {
            v = maximum - v;
        }

        if planar {
            *output
                .offset::<f64>(((i + start) * stride) as i32)
                .deref_mut() = v as f32;
        } else {
            *output.offset::<f64>((i + start) as i32).deref_mut() = v as f32;
        }
    }

    if !extra_first {
        output += extra * mem::size_of::<u16>() as u32;
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            (n_chan - 1) as usize * mem::size_of::<f32>(),
        );
        *swap1.deref_mut() = v;
    }

    if planar {
        output + mem::size_of::<f32>()
    } else {
        output + (n_chan + extra) * mem::size_of::<f32>() as u32
    }
}

unsafe fn pack_floats_from_float(
    info: &Transform,
    w_out: &mut [f32],
    output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.output_format) != 0;

    let maximum = if is_ink_space(info.output_format) {
        100.
    } else {
        1.
    };
    let mut swap1 = output;
    let mut v: f64 = 0.;
    let mut start = 0;

    let stride = stride / pixel_size(info.output_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };
        v = w_out[index as usize] as f64 * maximum;

        if reverse {
            v = maximum - v;
        }

        if planar {
            *output
                .offset::<f32>(((i + start) * stride) as i32)
                .deref_mut() = v as f32;
        } else {
            *output.offset::<f32>((i + start) as i32).deref_mut() = v as f32;
        }
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            (n_chan - 1) as usize * mem::size_of::<f32>(),
        );
        *swap1.deref_mut() = v;
    }

    if planar {
        output + mem::size_of::<f32>()
    } else {
        output + (n_chan + extra) * mem::size_of::<f32>() as u32
    }
}

unsafe fn pack_doubles_from_float(
    info: &Transform,
    w_out: &mut [f32],
    output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.output_format) != 0;

    let maximum = if is_ink_space(info.output_format) {
        100.
    } else {
        1.
    };
    let mut swap1 = output;
    let mut v: f64 = 0.;
    let mut start = 0;

    let stride = stride / pixel_size(info.output_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };
        v = w_out[index as usize] as f64 * maximum;

        if reverse {
            v = maximum - v;
        }

        if planar {
            *output
                .offset::<f64>(((i + start) * stride) as i32)
                .deref_mut() = v;
        } else {
            *output.offset::<f64>((i + start) as i32).deref_mut() = v;
        }
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            (n_chan - 1) as usize * mem::size_of::<f64>(),
        );
        *swap1.deref_mut() = v;
    }

    if planar {
        output + mem::size_of::<f64>()
    } else {
        output + (n_chan + extra) * mem::size_of::<f64>() as u32
    }
}

unsafe fn pack_lab_float_from_float(
    info: &Transform,
    w_out: &mut [f32],
    mut output: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.output_format) != 0 {
        let stride = stride / pixel_size(info.output_format);
        *output.deref_mut::<f32>() = w_out[0] * 100.;
        *output.offset::<f32>(stride as i32).deref_mut::<f32>() = w_out[1] * 255. - 128.;
        *output.offset::<f32>(stride as i32 * 2).deref_mut::<f32>() = w_out[2] * 255. - 128.;
        output + mem::size_of::<f32>()
    } else {
        *output.deref_mut::<f32>() = w_out[0] * 100.;
        *output.offset::<f32>(1).deref_mut::<f32>() = w_out[1] * 255. - 128.;
        *output.offset::<f32>(2).deref_mut::<f32>() = w_out[2] * 255. - 128.;
        output + (3 + t_extra!(info.output_format)) + mem::size_of::<f32>()
    }
}

unsafe fn pack_lab_double_from_float(
    info: &Transform,
    w_out: &mut [f32],
    mut output: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.output_format) != 0 {
        let stride = stride / pixel_size(info.output_format);
        *output.deref_mut::<f64>() = w_out[0] as f64 * 100.;
        *output.offset::<f64>(stride as i32).deref_mut::<f64>() = w_out[1] as f64 * 255. - 128.;
        *output.offset::<f64>(stride as i32 * 2).deref_mut::<f64>() = w_out[2] as f64 * 255. - 128.;
        output + mem::size_of::<f64>()
    } else {
        *output.deref_mut::<f64>() = w_out[0] as f64 * 100.;
        *output.offset::<f64>(1).deref_mut::<f64>() = w_out[1] as f64 * 255. - 128.;
        *output.offset::<f64>(2).deref_mut::<f64>() = w_out[2] as f64 * 255. - 128.;
        output + (3 + t_extra!(info.output_format)) + mem::size_of::<f64>()
    }
}

/// From 0..1 range to 0..MAX_ENCODEABLE_XYZ
unsafe fn pack_xyz_float_from_float(
    info: &Transform,
    w_out: &mut [f32],
    mut output: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.output_format) != 0 {
        let stride = stride / pixel_size(info.output_format);
        *output.deref_mut::<f32>() = w_out[0] * MAX_ENCODEABLE_XYZ as f32;
        *output.offset::<f32>(stride as i32).deref_mut::<f32>() =
            w_out[1] * MAX_ENCODEABLE_XYZ as f32;
        *output.offset::<f32>(stride as i32 * 2).deref_mut::<f32>() =
            w_out[2] * MAX_ENCODEABLE_XYZ as f32;
        output + mem::size_of::<f32>()
    } else {
        *output.deref_mut::<f32>() = w_out[0] * MAX_ENCODEABLE_XYZ as f32;
        *output.offset::<f32>(1).deref_mut::<f32>() = w_out[1] * MAX_ENCODEABLE_XYZ as f32;
        *output.offset::<f32>(2).deref_mut::<f32>() = w_out[2] * MAX_ENCODEABLE_XYZ as f32;
        output + (3 + t_extra!(info.output_format)) + mem::size_of::<f32>()
    }
}

/// From 0..1 range to 0..MAX_ENCODEABLE_XYZ
unsafe fn pack_xyz_double_from_float(
    info: &Transform,
    w_out: &mut [f32],
    mut output: Accum,
    stride: u32,
) -> Accum {
    if t_planar!(info.output_format) != 0 {
        let stride = stride / pixel_size(info.output_format);
        *output.deref_mut::<f64>() = w_out[0] as f64 * MAX_ENCODEABLE_XYZ;
        *output.offset::<f64>(stride as i32).deref_mut::<f64>() =
            w_out[1] as f64 * MAX_ENCODEABLE_XYZ;
        *output.offset::<f64>(stride as i32 * 2).deref_mut::<f64>() =
            w_out[2] as f64 * MAX_ENCODEABLE_XYZ;
        output + mem::size_of::<f64>()
    } else {
        *output.deref_mut::<f64>() = w_out[0] as f64 * MAX_ENCODEABLE_XYZ;
        *output.offset::<f64>(1).deref_mut::<f64>() = w_out[1] as f64 * MAX_ENCODEABLE_XYZ;
        *output.offset::<f64>(2).deref_mut::<f64>() = w_out[2] as f64 * MAX_ENCODEABLE_XYZ;
        output + (3 + t_extra!(info.output_format)) + mem::size_of::<f64>()
    }
}

/// Decodes an stream of half floats to wIn[] described by input format
unsafe fn unroll_half_to_16(
    info: &Transform,
    w_in: &mut [u16],
    accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.input_format) != 0;

    let mut v: f32;
    let mut start = 0;

    let maximum: f32 = if is_ink_space(info.input_format) {
        655.35
    } else {
        65535.
    };

    // TODO: check if this is supposed to be info.output_format (as in the original code) or if it should
    // actually be input_format
    let stride = stride / pixel_size(info.output_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        if planar {
            v = half_to_float(
                accum
                    .offset::<u16>(((i + start) * stride) as i32)
                    .deref_as::<u16>(),
            );
        } else {
            v = half_to_float(accum.offset::<u16>((i + start) as i32).deref_as::<u16>());
        }

        if reverse {
            v = maximum - v;
        }

        w_in[index as usize] = quick_saturate_word((v * maximum).into());
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    if planar {
        accum + mem::size_of::<u16>()
    } else {
        accum + (n_chan + extra) * mem::size_of::<u16>() as u32
    }
}

/// Decodes an stream of half floats to wIn[] described by input format
unsafe fn unroll_half_to_float(
    info: &Transform,
    w_in: &mut [f32],
    accum: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.input_format);
    let do_swap = t_do_swap!(info.input_format) != 0;
    let reverse = t_flavor!(info.input_format) != 0;
    let swap_first = t_swap_first!(info.input_format) != 0;
    let extra = t_extra!(info.input_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.input_format) != 0;

    let mut v: f32;
    let mut start = 0;
    let maximum = if is_ink_space(info.input_format) {
        100.
    } else {
        1.
    };

    // TODO: see same section above (unroll_half_to_16)
    let stride = stride / pixel_size(info.output_format);

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        if planar {
            v = half_to_float(
                accum
                    .offset::<u16>(((i + start) * stride) as i32)
                    .deref_as::<u16>(),
            );
        } else {
            v = half_to_float(accum.offset::<u16>((i + start) as i32).deref_as::<u16>());
        }

        v /= maximum;

        w_in[index as usize] = if reverse { 1. - v } else { v };
    }

    if extra == 0 && swap_first {
        let tmp = w_in[0];
        memmove(
            &mut w_in[0] as *mut _ as *mut _,
            &w_in[1] as *const _ as *const _,
            (n_chan - 1) as usize * mem::size_of::<f32>(),
        );
        w_in[n_chan as usize - 1] = tmp;
    }

    if planar {
        accum + mem::size_of::<u16>()
    } else {
        accum + (n_chan + extra) * mem::size_of::<u16>() as u32
    }
}

unsafe fn pack_half_from_16(
    info: &Transform,
    w_out: &mut [u16],
    output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.output_format) != 0;

    let mut v: f32 = 0.;
    let maximum = if is_ink_space(info.output_format) {
        655.35
    } else {
        65535.
    };
    let mut swap1 = output;
    let mut start = 0;

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        v = w_out[index as usize] as f32 / maximum;

        if reverse {
            v = maximum - v;
        }

        if planar {
            *output
                .offset::<u16>(((i + start) * stride) as i32)
                .deref_mut() = float_to_half(v);
        } else {
            *output.offset::<u16>((i + start) as i32).deref_mut() = float_to_half(v);
        }
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        *swap1.deref_mut() = float_to_half(v);
    }

    if planar {
        output + mem::size_of::<u16>()
    } else {
        output + (n_chan + extra) * mem::size_of::<u16>() as u32
    }
}

unsafe fn pack_half_from_float(
    info: &Transform,
    w_out: &mut [f32],
    output: Accum,
    stride: u32,
) -> Accum {
    let n_chan = t_channels!(info.output_format);
    let do_swap = t_do_swap!(info.output_format) != 0;
    let reverse = t_flavor!(info.output_format) != 0;
    let swap_first = t_swap_first!(info.output_format) != 0;
    let extra = t_extra!(info.output_format);
    let extra_first = do_swap ^ swap_first;
    let planar = t_planar!(info.output_format) != 0;

    let mut v: f32 = 0.;
    let maximum = if is_ink_space(info.output_format) {
        100.
    } else {
        1.
    };
    let mut swap1 = output;
    let mut start = 0;

    if extra_first {
        start = extra;
    }

    for i in 0..n_chan {
        let index = if do_swap { n_chan - i - 1 } else { i };

        v = w_out[index as usize] * maximum;

        if reverse {
            v = maximum - v;
        }

        if planar {
            *output
                .offset::<u16>(((i + start) * stride) as i32)
                .deref_mut() = float_to_half(v);
        } else {
            *output.offset::<u16>((i + start) as i32).deref_mut() = float_to_half(v);
        }
    }

    if extra == 0 && swap_first {
        memmove(
            (swap1 + 1_i32).as_mut() as *mut _,
            *swap1 as *const u8 as *const _,
            (n_chan - 1) as usize * mem::size_of::<u16>(),
        );
        *swap1.deref_mut() = float_to_half(v);
    }

    if planar {
        output + mem::size_of::<u16>()
    } else {
        output + (n_chan + extra) * mem::size_of::<u16>() as u32
    }
}

// colorimetric
const TYPE_XYZ_16: u32 = (colorspace_sh!((PixelType::XYZ as u32)) | channels_sh!(3) | bytes_sh!(2));
const TYPE_LAB_8: u32 = (colorspace_sh!((PixelType::Lab as u32)) | channels_sh!(3) | bytes_sh!(1));
const TYPE_LABV2_8: u32 =
    (colorspace_sh!((PixelType::LabV2 as u32)) | channels_sh!(3) | bytes_sh!(1));

const TYPE_ALAB_8: u32 = (colorspace_sh!((PixelType::Lab as u32))
    | channels_sh!(3)
    | bytes_sh!(1)
    | extra_sh!(1)
    | swap_first_sh!(1));
const TYPE_ALABV2_8: u32 = (colorspace_sh!((PixelType::LabV2 as u32))
    | channels_sh!(3)
    | bytes_sh!(1)
    | extra_sh!(1)
    | swap_first_sh!(1));
const TYPE_LAB_16: u32 = (colorspace_sh!((PixelType::Lab as u32)) | channels_sh!(3) | bytes_sh!(2));
const TYPE_LABV2_16: u32 =
    (colorspace_sh!((PixelType::LabV2 as u32)) | channels_sh!(3) | bytes_sh!(2));
const TYPE_YXY_16: u32 = (colorspace_sh!((PixelType::Yxy as u32)) | channels_sh!(3) | bytes_sh!(2));

// Float formatters.
const TYPE_XYZ_FLT: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::XYZ as u32)) | channels_sh!(3) | bytes_sh!(4));
const TYPE_LAB_FLT: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::Lab as u32)) | channels_sh!(3) | bytes_sh!(4));
const TYPE_LABA_FLT: u32 = (float_sh!(1)
    | colorspace_sh!((PixelType::Lab as u32))
    | extra_sh!(1)
    | channels_sh!(3)
    | bytes_sh!(4));
const TYPE_GRAY_FLT: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::Gray as u32)) | channels_sh!(1) | bytes_sh!(4));
const TYPE_RGB_FLT: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::RGB as u32)) | channels_sh!(3) | bytes_sh!(4));

// Floating point formatters.
// NOTE THAT 'BYTES' FIELD IS SET TO ZERO ON DBL because 8 bytes overflows the bitfield
const TYPE_XYZ_DBL: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::XYZ as u32)) | channels_sh!(3) | bytes_sh!(0));
pub(crate) const TYPE_LAB_DBL: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::Lab as u32)) | channels_sh!(3) | bytes_sh!(0));
const TYPE_GRAY_DBL: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::Gray as u32)) | channels_sh!(1) | bytes_sh!(0));
const TYPE_RGB_DBL: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::RGB as u32)) | channels_sh!(3) | bytes_sh!(0));
const TYPE_BGR_DBL: u32 = (float_sh!(1)
    | colorspace_sh!((PixelType::RGB as u32))
    | channels_sh!(3)
    | bytes_sh!(0)
    | do_swap_sh!(1));
const TYPE_CMYK_DBL: u32 =
    (float_sh!(1) | colorspace_sh!((PixelType::CMYK as u32)) | channels_sh!(4) | bytes_sh!(0));

const INPUT_FORMATTERS_16: [Formatters16; 43] = [
    Formatters16 {
        ty: TYPE_LAB_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_lab_double_to_16,
    },
    Formatters16 {
        ty: TYPE_XYZ_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_xyz_double_to_16,
    },
    Formatters16 {
        ty: TYPE_LAB_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_lab_float_to_16,
    },
    Formatters16 {
        ty: TYPE_XYZ_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_xyz_float_to_16,
    },
    Formatters16 {
        ty: TYPE_GRAY_DBL,
        mask: 0,
        frm: unroll_double_1_chan,
    },
    Formatters16 {
        ty: float_sh!(1) | bytes_sh!(0),
        mask: ANYCHANNELS | ANYPLANAR | ANYSWAPFIRST | ANYFLAVOR | ANYSWAP | ANYEXTRA | ANYSPACE,
        frm: unroll_double_to_16,
    },
    Formatters16 {
        ty: float_sh!(1) | bytes_sh!(4),
        mask: ANYCHANNELS | ANYPLANAR | ANYSWAPFIRST | ANYFLAVOR | ANYSWAP | ANYEXTRA | ANYSPACE,
        frm: unroll_float_to_16,
    },
    Formatters16 {
        ty: float_sh!(1) | bytes_sh!(2),
        mask: ANYCHANNELS | ANYPLANAR | ANYSWAPFIRST | ANYFLAVOR | ANYEXTRA | ANYSWAP | ANYSPACE,
        frm: unroll_half_to_16,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1),
        mask: ANYSPACE,
        frm: unroll_1_byte,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1) | extra_sh!(1),
        mask: ANYSPACE,
        frm: unroll_1_byte_skip_1,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1) | extra_sh!(2),
        mask: ANYSPACE,
        frm: unroll_1_byte_skip_2,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: unroll_1_byte_reversed,
    },
    Formatters16 {
        ty: colorspace_sh!((PixelType::MCH2 as u32)) | channels_sh!(2) | bytes_sh!(1),
        mask: 0,
        frm: unroll_2_bytes,
    },
    Formatters16 {
        ty: TYPE_LABV2_8,
        mask: 0,
        frm: unroll_labv2_8,
    },
    Formatters16 {
        ty: TYPE_ALABV2_8,
        mask: 0,
        frm: unroll_a_labv2_8,
    },
    Formatters16 {
        ty: TYPE_LABV2_16,
        mask: 0,
        frm: unroll_labv2_16,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_bytes,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_bytes_swap,
    },
    Formatters16 {
        ty: channels_sh!(3) | extra_sh!(1) | bytes_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_bytes_skip_1_swap,
    },
    Formatters16 {
        ty: channels_sh!(3) | extra_sh!(1) | bytes_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_bytes_skip_1_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(3) | extra_sh!(1) | bytes_sh!(1) | do_swap_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_bytes_skip_1_swap_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_bytes,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_bytes_reverse,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_bytes_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_bytes_swap,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | do_swap_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_bytes_swap_swap_first,
    },
    Formatters16 {
        ty: bytes_sh!(1) | planar_sh!(1),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: unroll_planar_bytes,
    },
    Formatters16 {
        ty: bytes_sh!(1),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: unroll_chunky_bytes,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: unroll_1_word,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: unroll_1_word_reversed,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2) | extra_sh!(3),
        mask: ANYSPACE,
        frm: unroll_1_word_skip_3,
    },
    Formatters16 {
        ty: channels_sh!(2) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: unroll_2_words,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: unroll_3_words,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: unroll_4_words,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_words_swap,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | extra_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_words_skip_1_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | extra_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: unroll_3_words_skip_1_swap,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_words_reverse,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_words_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_words_swap,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2) | do_swap_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: unroll_4_words_swap_swap_first,
    },
    Formatters16 {
        ty: bytes_sh!(2) | planar_sh!(1),
        mask: ANYFLAVOR | ANYSWAP | ANYENDIAN | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: unroll_planar_words,
    },
    Formatters16 {
        ty: bytes_sh!(2),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYENDIAN | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: unroll_any_words,
    },
];

const INPUT_FORMATTERS_FLOAT: [FormattersFloat; 7] = [
    FormattersFloat {
        ty: TYPE_LAB_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_lab_double_to_float,
    },
    FormattersFloat {
        ty: TYPE_LAB_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_lab_float_to_float,
    },
    FormattersFloat {
        ty: TYPE_XYZ_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_xyz_double_to_float,
    },
    FormattersFloat {
        ty: TYPE_XYZ_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: unroll_xyz_float_to_float,
    },
    FormattersFloat {
        ty: float_sh!(1) | bytes_sh!(4),
        mask: ANYPLANAR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: unroll_floats_to_float,
    },
    FormattersFloat {
        ty: float_sh!(1) | bytes_sh!(0),
        mask: ANYPLANAR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: unroll_doubles_to_float,
    },
    FormattersFloat {
        ty: float_sh!(1) | bytes_sh!(2),
        mask: ANYPLANAR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: unroll_half_to_float,
    },
];

pub const PACK_FLAGS_16BITS: u32 = 0x0000;
pub const PACK_FLAGS_FLOAT: u32 = 0x0001;

/// Encapsulates a formatter function.
#[derive(Clone, Copy)]
pub enum Formatter {
    Sixteen(Formatter16),
    Float(FormatterFloat),
    None,
}

impl Formatter {
    pub fn unwrap_16(self) -> Formatter16 {
        match self {
            Formatter::Sixteen(f) => f,
            _ => panic!("Unwrapping Formatter; expected 16"),
        }
    }

    pub fn unwrap_float(self) -> FormatterFloat {
        match self {
            Formatter::Float(f) => f,
            _ => panic!("Unwrapping Formatter; expected Float"),
        }
    }
}

impl fmt::Debug for Formatter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Formatter::Sixteen(_) => "Formatter16(fn)",
                Formatter::Float(_) => "FormatterFloat(fn)",
                Formatter::None => "FormatterNone",
            }
        )
    }
}

/// Bit fields set to one in the mask are not compared
fn stock_input_formatter(dw_input: u32, dw_flags: u32) -> Option<Formatter> {
    match dw_flags {
        PACK_FLAGS_16BITS => {
            for formatter in INPUT_FORMATTERS_16.iter() {
                if dw_input & !formatter.mask == formatter.ty {
                    return Some(Formatter::Sixteen(formatter.frm));
                }
            }
            None
        }
        PACK_FLAGS_FLOAT => {
            for formatter in INPUT_FORMATTERS_FLOAT.iter() {
                if dw_input & !formatter.mask == formatter.ty {
                    return Some(Formatter::Float(formatter.frm));
                }
            }
            None
        }
        _ => None,
    }
}

const OUTPUT_FORMATTERS_16: [Formatters16; 55] = [
    Formatters16 {
        ty: TYPE_LAB_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_lab_double_from_16,
    },
    Formatters16 {
        ty: TYPE_XYZ_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_xyz_double_from_16,
    },
    Formatters16 {
        ty: TYPE_LAB_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_lab_float_from_16,
    },
    Formatters16 {
        ty: TYPE_XYZ_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_xyz_float_from_16,
    },
    Formatters16 {
        ty: float_sh!(1) | bytes_sh!(0),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYCHANNELS | ANYPLANAR | ANYEXTRA | ANYSPACE,
        frm: pack_double_from_16,
    },
    Formatters16 {
        ty: float_sh!(1) | bytes_sh!(4),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYCHANNELS | ANYPLANAR | ANYEXTRA | ANYSPACE,
        frm: pack_float_from_16,
    },
    Formatters16 {
        ty: float_sh!(1) | bytes_sh!(2),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYCHANNELS | ANYPLANAR | ANYEXTRA | ANYSPACE,
        frm: pack_half_from_16,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_byte,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1) | extra_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_byte_skip_1,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1) | extra_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_byte_skip_1_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(1) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_byte_reversed,
    },
    Formatters16 {
        ty: TYPE_LABV2_8,
        mask: 0,
        frm: pack_labv2_8,
    },
    Formatters16 {
        ty: TYPE_ALABV2_8,
        mask: 0,
        frm: pack_a_labv2_8,
    },
    Formatters16 {
        ty: TYPE_LABV2_16,
        mask: 0,
        frm: pack_labv2_16,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | optimized_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_optimized,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | extra_sh!(1) | optimized_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1_optimized,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | extra_sh!(1) | swap_first_sh!(1) | optimized_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1_swap_first_optimized,
    },
    Formatters16 {
        ty: channels_sh!(3)
            | bytes_sh!(1)
            | extra_sh!(1)
            | do_swap_sh!(1)
            | swap_first_sh!(1)
            | optimized_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1_swap_swap_first_optimized,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | do_swap_sh!(1) | extra_sh!(1) | optimized_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1_swap_optimized,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | do_swap_sh!(1) | optimized_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_swap_optimized,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | extra_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | extra_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | extra_sh!(1) | do_swap_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1_swap_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | do_swap_sh!(1) | extra_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_and_skip_1_swap,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_bytes_swap,
    },
    Formatters16 {
        ty: channels_sh!(6) | bytes_sh!(1),
        mask: ANYSPACE,
        frm: pack_6_bytes,
    },
    Formatters16 {
        ty: channels_sh!(6) | bytes_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: pack_6_bytes_swap,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_bytes,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_bytes_reverse,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_bytes_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_bytes_swap,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(1) | do_swap_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_bytes_swap_swap_first,
    },
    Formatters16 {
        ty: bytes_sh!(1),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: pack_any_bytes,
    },
    Formatters16 {
        ty: bytes_sh!(1) | planar_sh!(1),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: pack_planar_bytes,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: pack_1_word,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2) | extra_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_word_skip_1,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2) | extra_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_word_skip_1_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_word_reversed,
    },
    Formatters16 {
        ty: channels_sh!(1) | bytes_sh!(2) | endian16_sh!(1),
        mask: ANYSPACE,
        frm: pack_1_word_big_endian,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: pack_3_words,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_words_swap,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | endian16_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_words_big_endian,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | extra_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_words_and_skip_1,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | extra_sh!(1) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_words_and_skip_1_swap,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | extra_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_words_and_skip_1_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(3) | bytes_sh!(2) | extra_sh!(1) | do_swap_sh!(1) | swap_first_sh!(1),
        mask: ANYSPACE,
        frm: pack_3_words_and_skip_1_swap_swap_first,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: pack_4_words,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2) | flavor_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_words_reverse,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_words_swap,
    },
    Formatters16 {
        ty: channels_sh!(4) | bytes_sh!(2) | endian16_sh!(1),
        mask: ANYSPACE,
        frm: pack_4_words_big_endian,
    },
    Formatters16 {
        ty: channels_sh!(6) | bytes_sh!(2),
        mask: ANYSPACE,
        frm: pack_6_words,
    },
    Formatters16 {
        ty: channels_sh!(6) | bytes_sh!(2) | do_swap_sh!(1),
        mask: ANYSPACE,
        frm: pack_6_words_swap,
    },
    Formatters16 {
        ty: bytes_sh!(2) | planar_sh!(1),
        mask: ANYFLAVOR | ANYENDIAN | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: pack_planar_words,
    },
    Formatters16 {
        ty: bytes_sh!(2),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYENDIAN | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: pack_any_words,
    },
];

const OUTPUT_FORMATTERS_FLOAT: [FormattersFloat; 7] = [
    FormattersFloat {
        ty: TYPE_LAB_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_lab_float_from_float,
    },
    FormattersFloat {
        ty: TYPE_XYZ_FLT,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_xyz_float_from_float,
    },
    FormattersFloat {
        ty: TYPE_LAB_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_lab_double_from_float,
    },
    FormattersFloat {
        ty: TYPE_XYZ_DBL,
        mask: ANYPLANAR | ANYEXTRA,
        frm: pack_xyz_double_from_float,
    },
    FormattersFloat {
        ty: float_sh!(1) | bytes_sh!(4),
        mask: ANYPLANAR | ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: pack_floats_from_float,
    },
    FormattersFloat {
        ty: float_sh!(1) | bytes_sh!(0),
        mask: ANYPLANAR | ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: pack_doubles_from_float,
    },
    FormattersFloat {
        ty: float_sh!(1) | bytes_sh!(2),
        mask: ANYFLAVOR | ANYSWAPFIRST | ANYSWAP | ANYEXTRA | ANYCHANNELS | ANYSPACE,
        frm: pack_half_from_float,
    },
];

/// Bit fields set to one in the mask are not compared
fn stock_output_formatter(dw_input: u32, dw_flags: u32) -> Option<Formatter> {
    // Optimization is only a hint
    let dw_input = dw_input & !optimized_sh!(1);

    match dw_flags {
        PACK_FLAGS_16BITS => {
            for formatter in OUTPUT_FORMATTERS_16.iter() {
                if dw_input & !formatter.mask == formatter.ty {
                    return Some(Formatter::Sixteen(formatter.frm));
                }
            }
            None
        }
        PACK_FLAGS_FLOAT => {
            for formatter in OUTPUT_FORMATTERS_FLOAT.iter() {
                if dw_input & !formatter.mask == formatter.ty {
                    return Some(Formatter::Float(formatter.frm));
                }
            }
            None
        }
        _ => None,
    }
}

/// Formatting directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatterDir {
    Input = 0,
    Output = 1,
}

/// Returns a formatter for the given parameters.
pub fn formatter(ty: u32, dir: FormatterDir, flags: u32) -> Option<Formatter> {
    // no plugins

    match dir {
        FormatterDir::Input => stock_input_formatter(ty, flags),
        FormatterDir::Output => stock_output_formatter(ty, flags),
    }
}

pub(crate) fn formatter_is_float(t: u32) -> bool {
    t_float!(t) != 0
}

pub(crate) fn formatter_is_8_bit(t: u32) -> bool {
    t_bytes!(t) == 1
}

/// Builds a suitable formatter for the color space of this profile
pub fn formatter_for_color_space_of_profile(
    profile: &Profile,
    bytes: u32,
    l_is_float: bool,
) -> u32 {
    let color_space = profile.color_space;
    let cs_bits = lcms_color_space(color_space) as u32;
    let n_output_chans = color_space.channels();
    let float = if l_is_float { 1 } else { 0 };

    // create a fake formatter for result
    float_sh!(float) | colorspace_sh!(cs_bits) | bytes_sh!(bytes) | channels_sh!(n_output_chans)
}

/// Builds a suitable formatter for the PCS color space of this profile
pub fn formatter_for_pcs_of_profile(profile: &Profile, bytes: u32, l_is_float: bool) -> u32 {
    let color_space = profile.pcs;
    let cs_bits = lcms_color_space(color_space) as u32;
    let n_output_chans = color_space.channels();
    let float = if l_is_float { 1 } else { 0 };

    // create a fake formatter for result
    float_sh!(float) | colorspace_sh!(cs_bits) | bytes_sh!(bytes) | channels_sh!(n_output_chans)
}
