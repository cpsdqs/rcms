//! Color management library.
//!
//! Heavily based on Little CMS.

#[macro_use]
mod util;

mod black_point;
pub mod color;
pub mod fixed;
pub mod link;
pub mod pipeline;
pub mod profile;
pub mod tone_curve;

pub use profile::IccProfile;
pub use tone_curve::ToneCurve;
