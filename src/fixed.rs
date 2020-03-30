//! Fixed-point types.

/// Signed 15.16 fixed-point number.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct s15f16(pub i32);

impl s15f16 {
    pub fn to_f64(self) -> f64 {
        let sign = self.0.signum() as f64;
        let value = self.0.abs();

        let whole = (value >> 16) as u16 & 0xffff;
        let frac_part = value as u16 & 0xffff;
        let mid = frac_part as f64 / 65536.;
        let floater = whole as f64 + mid;

        sign * floater
    }

    pub fn from_f64(this: f64) -> Self {
        let sign = this.signum() as i32;
        let value = this.abs();

        let whole = value.floor();
        let frac_part = value - whole;

        s15f16(sign * (((whole as i32 & 0x7FFF) << 16) | (((frac_part * 65536.) as i32) & 0xFFFF)))
    }
}

/// Unsigned 16.16 fixed-point number.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct u16f16(pub u32);

impl u16f16 {
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / 65536.
    }
    pub fn from_f64(this: f64) -> Self {
        u16f16((this * 65536.) as u32)
    }
}
