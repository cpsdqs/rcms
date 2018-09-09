// Format of pixel is defined by one cmsUInt32Number, using bit fields as follows
//
//                               2                1          0
//                          3 2 10987 6 5 4 3 2 1 098 7654 321
//                          A O TTTTT U Y F P X S EEE CCCC BBB
//
//            A: Floating point -- With this flag we can differentiate 16 bits as float and as int
//            O: Optimized -- previous optimization already returns the final 8-bit value
//            T: Pixeltype
//            F: Flavor  0=MinIsBlack(Chocolate) 1=MinIsWhite(Vanilla)
//            P: Planar? 0=Chunky, 1=Planar
//            X: swap 16 bps endianness?
//            S: Do swap? ie, BGR, KYMC
//            E: Extra samples
//            C: Channels (Samples per pixel)
//            B: bytes per sample
//            Y: Swap first - changes ABGR to BGRA and KCMY to CMYK

macro_rules! float_sh {
    ($a:tt) => {
        (($a) << 22)
    };
}
macro_rules! optimized_sh {
    ($s:tt) => {
        (($s) << 21)
    };
}
macro_rules! colorspace_sh {
    ($s:tt) => {
        (($s) << 16)
    };
}
macro_rules! swap_first_sh {
    ($s:tt) => {
        (($s) << 14)
    };
}
macro_rules! flavor_sh {
    ($s:tt) => {
        (($s) << 13)
    };
}
macro_rules! planar_sh {
    ($p:tt) => {
        (($p) << 12)
    };
}
macro_rules! endian16_sh {
    ($e:tt) => {
        (($e) << 11)
    };
}
macro_rules! do_swap_sh {
    ($e:tt) => {
        (($e) << 10)
    };
}
macro_rules! extra_sh {
    ($e:tt) => {
        (($e) << 7)
    };
}
macro_rules! channels_sh {
    ($c:tt) => {
        (($c) << 3)
    };
}
macro_rules! bytes_sh {
    ($b:tt) => {
        ($b)
    };
}

// These macros unpack format specifiers into integers.
macro_rules! t_float {
    ($a:expr) => {
        ($a >> 22) & 1
    };
}
/*
macro_rules! t_optimized {
    ($o:expr) => {
        ($o >> 21) & 1
    };
}
*/
macro_rules! t_colorspace {
    ($s:expr) => {
        ($s >> 16) & 31
    };
}
macro_rules! t_swap_first {
    ($s:expr) => {
        ($s >> 14) & 1
    };
}
macro_rules! t_flavor {
    ($s:expr) => {
        ($s >> 13) & 1
    };
}
macro_rules! t_planar {
    ($p:expr) => {
        ($p >> 12) & 1
    };
}
macro_rules! t_endian16 {
    ($e:expr) => {
        ($e >> 11) & 1
    };
}
macro_rules! t_do_swap {
    ($e:expr) => {
        ($e >> 10) & 1
    };
}
macro_rules! t_extra {
    ($e:expr) => {
        ($e >> 7) & 7
    };
}
macro_rules! t_channels {
    ($c:expr) => {
        ($c >> 3) & 15
    };
}
macro_rules! t_bytes {
    ($b:expr) => {
        $b & 7
    };
}

// A fast way to convert from/to 16 <-> 8 bits
macro_rules! from_8_to_16 {
    ($rgb:expr) => {
        ((($rgb as u16) << 8) | ($rgb as u16)) as u16
    };
}
macro_rules! from_16_to_8 {
    ($rgb:expr) => {
        (((($rgb as u32) * 65281 + 8388608) >> 24) & 0xFF) as u8
    };
}
