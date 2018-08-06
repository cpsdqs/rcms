use S15Fixed16;

pub fn s15fixed16_to_double(fix32: S15Fixed16) -> f64 {
    let sign = fix32.signum() as f64;
    let fix32 = fix32.abs();

    let whole = (fix32 >> 16) as u16 & 0xffff;
    let frac_part = fix32 as u16 & 0xffff;
    let mid = frac_part as f64 / 65536.;
    let floater = whole as f64 + mid;

    sign * floater
}

// TODO
