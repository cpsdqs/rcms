#[repr(C)]
union QuickFloorUnion {
    val: f64,
    halves: [i32; 2],
}

/// Fast floor conversion logic. Thanks to Sree Kotay and Stuart Nixon
///
/// Note than this only works in the range ..-32767...+32767 because
/// mantissa is interpreted as 15.16 fixed point.
/// The union is to avoid pointer aliasing overoptimization.
pub fn quick_floor(val: f64) -> i32 {
    // 2^36 * 1.5, (52-16=36) uses limited precision to floor
    let lcms_double2fixmagic = 68719476736.0 * 1.5;

    let temp = QuickFloorUnion {
        val: val + lcms_double2fixmagic,
    };

    if cfg!(target_endian = "big") {
        unsafe { temp.halves[1] >> 16 }
    } else {
        unsafe { temp.halves[0] >> 16 }
    }
}

/// Fast floor restricted to 0..65535.0
pub fn quick_floor_word(d: f64) -> u16 {
    (quick_floor(d - 32767.) as u16).wrapping_add(32767)
}

// Floor to word, taking care of saturation
pub fn quick_saturate_word(d: f64) -> u16 {
    let d = d + 0.5;
    if d <= 0. {
        0
    } else if d >= 65535. {
        0xffff
    } else {
        quick_floor_word(d)
    }
}

pub const MATRIX_DET_TOLERANCE: f64 = 0.0001;
