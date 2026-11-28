mod percentile;

use std::slice;

/// # Safety
/// Values and counts must point to readable, aligned arrays. Scratch and output
/// must be writable, aligned, disjoint from each other and from both inputs.
#[no_mangle]
pub unsafe extern "C" fn select_depth_samples(
    values: *const u16,
    value_count: usize,
    counts: *const i64,
    row_count: usize,
    percentile: f64,
    scratch: *mut u16,
    scratch_count: usize,
    output: *mut i64,
) -> bool {
    percentile::select_samples(
        slice::from_raw_parts(values, value_count),
        slice::from_raw_parts(counts, row_count),
        percentile,
        slice::from_raw_parts_mut(scratch, scratch_count),
        slice::from_raw_parts_mut(output, row_count),
        f64::from,
    )
}
