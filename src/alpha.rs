use half::{float_to_half, half_to_float};
use internal::{quick_floor_word, quick_saturate_word};
use std::{mem, ptr};
use transform_tmp::{Stride, Transform, TransformFlags};

/// Floors to byte, taking care of saturation.
pub(crate) fn quick_saturate_byte(d: f64) -> u8 {
    let d = d + 0.5;
    if d <= 0. {
        0
    } else if d >= 255. {
        255
    } else {
        quick_floor_word(d) as u8
    }
}

/// Returns the size in bytes of a given formatter.
pub(crate) fn true_byte_size(format: u32) -> u32 {
    let fmt_bytes = t_bytes!(format);
    if fmt_bytes == 0 {
        // for double, the t_bytes field returns zero
        mem::size_of::<f64>() as u32
    } else {
        // otherwise, it's already correct for all formats
        fmt_bytes
    }
}

/// Returns the position (x or y) of the formatter in the table of functions.
fn formatter_pos(frm: u32) -> Option<AlphaType> {
    let b = t_bytes!(frm);

    if t_float!(frm) != 0 {
        match b {
            0 => Some(AlphaType::Double),
            2 => Some(AlphaType::Half),
            4 => Some(AlphaType::Float),
            _ => None,
        }
    } else {
        match b {
            2 => Some(AlphaType::Sixteen),
            1 => Some(AlphaType::Eight),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AlphaType {
    Eight = 0,
    Sixteen = 1,
    Half = 2,
    Float = 3,
    Double = 4,
}

/// An alpha-to-alpha function formatter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AlphaFormatter {
    from: AlphaType,
    to: AlphaType,
}

impl AlphaFormatter {
    // TODO: see if raw pointers are necessary
    unsafe fn convert(self, src: *const (), dst: *mut ()) {
        match self.from {
            AlphaType::Eight => {
                let src = *(src as *const u8);
                match self.to {
                    AlphaType::Eight => *(dst as *mut u8) = src,
                    AlphaType::Sixteen => *(dst as *mut u16) = from_8_to_16!(src),
                    AlphaType::Half => *(dst as *mut u16) = float_to_half(src as f32 / 255.),
                    AlphaType::Float => *(dst as *mut f32) = (src as f32) / 255.,
                    AlphaType::Double => *(dst as *mut f64) = (src as f64) / 255.,
                }
            }
            AlphaType::Sixteen => {
                let src = *(src as *const u16);
                match self.to {
                    AlphaType::Eight => *(dst as *mut u8) = from_16_to_8!(src),
                    AlphaType::Sixteen => *(dst as *mut u16) = src,
                    AlphaType::Half => *(dst as *mut u16) = float_to_half(src as f32 / 65535.),
                    AlphaType::Float => *(dst as *mut f32) = src as f32 / 65535.,
                    AlphaType::Double => *(dst as *mut f64) = src as f64 / 65535.,
                }
            }
            AlphaType::Half => {
                let src = *(src as *const u16);
                match self.to {
                    AlphaType::Eight => {
                        *(dst as *mut u8) = quick_saturate_byte(half_to_float(src) as f64 * 255.)
                    }
                    AlphaType::Sixteen => {
                        *(dst as *mut u16) = quick_saturate_word(half_to_float(src) as f64 * 65535.)
                    }
                    AlphaType::Half => *(dst as *mut u16) = src,
                    AlphaType::Float => *(dst as *mut f32) = half_to_float(src),
                    AlphaType::Double => *(dst as *mut f64) = half_to_float(src) as f64,
                }
            }
            AlphaType::Float => {
                let src = *(src as *const f32);
                match self.to {
                    AlphaType::Eight => *(dst as *mut u8) = quick_saturate_byte(src as f64 * 255.),
                    AlphaType::Sixteen => {
                        *(dst as *mut u16) = quick_saturate_word(src as f64 * 65535.)
                    }
                    AlphaType::Half => *(dst as *mut u16) = float_to_half(src),
                    AlphaType::Float => *(dst as *mut f32) = src,
                    AlphaType::Double => *(dst as *mut f64) = src as f64,
                }
            }
            AlphaType::Double => {
                let src = *(src as *const f64);
                match self.to {
                    AlphaType::Eight => *(dst as *mut u8) = quick_saturate_byte(src * 255.),
                    AlphaType::Sixteen => *(dst as *mut u16) = quick_saturate_word(src * 65535.),
                    AlphaType::Half => *(dst as *mut u16) = float_to_half(src as f32),
                    AlphaType::Float => *(dst as *mut f32) = src as f32,
                    AlphaType::Double => *(dst as *mut f64) = src,
                }
            }
        }
    }
}

// Obtains an alpha-to-alpha function formatter.
fn formatter_alpha(p_in: u32, out: u32) -> Option<AlphaFormatter> {
    let in_n = formatter_pos(p_in);
    let out_n = formatter_pos(out);

    if let (Some(in_n), Some(out_n)) = (in_n, out_n) {
        Some(AlphaFormatter {
            from: in_n,
            to: out_n,
        })
    } else {
        None
    }
}

pub const MAX_CHANNELS: usize = 16;

/// Computes the distance from each component to the next one in bytes.
fn compute_increments_for_chunky(
    format: u32,
    component_starting_order: &mut [u32],
    component_pointer_increments: &mut [u32],
) {
    let extra = t_extra!(format as usize);
    let nchannels = t_channels!(format as usize);
    let total_chans = nchannels + extra;
    let channel_size = true_byte_size(format) as usize;
    let pixel_size = channel_size * total_chans;

    // Sanity check
    if total_chans <= 0 || total_chans >= MAX_CHANNELS {
        return;
    }

    let mut channels: [u32; MAX_CHANNELS] = [0; MAX_CHANNELS];

    // Separation is independent of starting point and only depends on channel size
    for i in 0..extra {
        component_pointer_increments[i] = pixel_size as u32;
    }

    // Handle do swap
    for i in 0..total_chans {
        if t_do_swap!(format) != 0 {
            channels[i] = (total_chans - i - 1) as u32;
        } else {
            channels[i] = i as u32;
        }
    }

    // Handle swap first (ROL of positions), example CMYK -> KCMY | 0123 -> 3012
    if t_swap_first!(format) != 0 && total_chans > 1 {
        let tmp = channels[0];
        for i in 0..(total_chans - 1) {
            channels[i] = channels[i + 1];
        }
        channels[total_chans - 1] = tmp;
    }

    // Handle size
    if channel_size > 1 {
        for i in 0..total_chans {
            channels[i] *= channel_size as u32;
        }
    }

    for i in 0..extra {
        component_starting_order[i] = channels[i + nchannels];
    }
}

/// On planar configurations, the distance is the stride added to any non-negative.
fn compute_increments_for_planar(
    format: u32,
    bytes_per_plane: u32,
    component_starting_order: &mut [u32],
    component_pointer_increments: &mut [u32],
) {
    let extra = t_extra!(format as usize);
    let nchannels = t_channels!(format as usize);
    let total_chans = nchannels + extra;
    let channel_size = true_byte_size(format);

    // Sanity check
    if total_chans <= 0 || total_chans >= MAX_CHANNELS {
        return;
    }

    let mut channels: [u32; MAX_CHANNELS] = [0; MAX_CHANNELS];

    // Separation is independent of starting point and only depends on channel size
    for i in 0..extra {
        component_pointer_increments[i] = channel_size;
    }

    // Handle do swap
    for i in 0..total_chans {
        if t_do_swap!(format) != 0 {
            channels[i] = (total_chans - i - 1) as u32;
        } else {
            channels[i] = i as u32;
        }
    }

    // Handle swap first (ROL of positions), example CMYK -> KCMY | 0123 -> 3012
    if t_swap_first!(format) != 0 && total_chans > 0 {
        let tmp = channels[0];
        for i in 0..(total_chans - 1) {
            channels[i] = channels[i + 1];
        }
        channels[total_chans - 1] = tmp;
    }

    // Handle size
    for i in 0..total_chans {
        channels[i] *= bytes_per_plane;
    }

    for i in 0..extra {
        component_starting_order[i] = channels[i + nchannels];
    }
}

/// Dispatcher for chunky and planar RGB.
fn compute_component_increments(
    format: u32,
    bytes_per_plane: u32,
    component_starting_order: &mut [u32],
    component_pointer_increments: &mut [u32],
) {
    if t_planar!(format) != 0 {
        compute_increments_for_planar(
            format,
            bytes_per_plane,
            component_starting_order,
            component_pointer_increments,
        );
    } else {
        compute_increments_for_chunky(
            format,
            component_starting_order,
            component_pointer_increments,
        );
    }
}

// Handles extra channels, copying alpha if requested by the flags.
pub(crate) unsafe fn handle_extra_channels(
    transform: &Transform,
    input: *const (),
    output: *mut (),
    width: usize,
    height: usize,
    stride: Stride,
) {
    // Make sure we need some copy
    if !transform
        .dw_original_flags
        .contains(TransformFlags::COPY_ALPHA)
    {
        return;
    }

    // Exit early if in-place color-management is occurring - no need to copy extra channels to themselves.
    if transform.input_format == transform.output_format && input == output {
        return;
    }

    // Make sure we have same number of alpha channels. If not, just return as this should be checked at transform creation time.
    let n_extra = t_extra!(transform.input_format) as usize;
    if n_extra != t_extra!(transform.output_format) as usize {
        return;
    }

    // Anything to do?
    if n_extra == 0 {
        return;
    }

    let mut source_starting_order = [0; 16];
    let mut source_increments = [0; 16];
    let mut dest_starting_order = [0; 16];
    let mut dest_increments = [0; 16];

    // Compute the increments
    compute_component_increments(
        transform.input_format,
        stride.bytes_per_plane_in,
        &mut source_starting_order,
        &mut source_increments,
    );
    compute_component_increments(
        transform.output_format,
        stride.bytes_per_plane_out,
        &mut dest_starting_order,
        &mut dest_increments,
    );

    // Check for conversions 8, 16, half, float, dbl
    let converter = match formatter_alpha(transform.input_format, transform.output_format) {
        Some(fmt) => fmt,
        None => return,
    };

    if n_extra == 1 {
        // Optimized routine for copying a single extra channel quickly
        let mut source_stride_increment = 0;
        let mut dest_stride_increment = 0;

        // The loop itself
        for _ in 0..height {
            // Prepare pointers for the loop
            let mut source_ptr =
                input.offset((source_starting_order[0] + source_stride_increment) as isize);
            let mut dest_ptr =
                output.offset((dest_starting_order[0] + dest_stride_increment) as isize);

            for _ in 0..width {
                converter.convert(source_ptr, dest_ptr);

                source_ptr = source_ptr.offset(source_increments[0] as isize);
                dest_ptr = dest_ptr.offset(dest_increments[0] as isize);
            }

            source_stride_increment += stride.bytes_per_plane_in;
            dest_stride_increment += stride.bytes_per_plane_out;
        }
    } else {
        // General case with more than one extra channel

        let mut source_ptr = [ptr::null(); 16];
        let mut dest_ptr = [ptr::null_mut(); 16];
        let mut source_stride_increments = [0; 16];
        let mut dest_stride_increments = [0; 16];

        // The loop itself
        for _ in 0..height {
            // Prepare pointers for the loop
            for i in 0..n_extra {
                source_ptr[i] =
                    input.offset((source_starting_order[i] + source_stride_increments[i]) as isize);
                dest_ptr[i] =
                    output.offset((dest_starting_order[i] + dest_stride_increments[i]) as isize);
            }

            for _ in 0..width {
                for i in 0..n_extra {
                    converter.convert(source_ptr[i], dest_ptr[i]);

                    source_ptr[i] = source_ptr[i].offset(source_increments[i] as isize);
                    source_ptr[i] = source_ptr[i].offset(dest_increments[i] as isize);
                }
            }

            for i in 0..n_extra {
                source_stride_increments[i] = stride.bytes_per_plane_in;
                dest_stride_increments[i] = stride.bytes_per_plane_out;
            }
        }
    }
}
