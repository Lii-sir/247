#![no_std]

/// Count foreground pixels and write the flat indices outside the rectangle.
///
/// # Safety
/// `mask` contains width * height bytes. The half-open rectangle is within
/// the image. `outside` can hold one index for every pixel outside it, and
/// `silver_count` points to a writable usize.
#[no_mangle]
pub unsafe extern "C" fn scan_silver_rectangle(
    mask: *const u8,
    width: usize,
    height: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    outside: *mut usize,
    silver_count: *mut usize,
) -> usize {
    let mut total_inside = 0;
    let mut total_outside = 0;
    for y in 0..height {
        let base = y * width;
        let (left, right) = if y >= y0 && y < y1 { (x0, x1) } else { (width, width) };
        for x in 0..left {
            if *mask.add(base + x) != 0 {
                outside.add(total_outside).write(base + x);
                total_outside += 1;
            }
        }
        for x in left..right {
            total_inside += usize::from(*mask.add(base + x) != 0);
        }
        for x in right..width {
            if *mask.add(base + x) != 0 {
                outside.add(total_outside).write(base + x);
                total_outside += 1;
            }
        }
    }
    silver_count.write(total_inside + total_outside);
    total_outside
}
