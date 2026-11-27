#![no_std]

use core::arch::x86_64::*;

// MSVC marks objects that use floating point. This DLL has no CRT dependency.
#[no_mangle]
static _fltused: i32 = 0;

#[no_mangle]
pub extern "C" fn affine_has_fma() -> bool {
    __cpuid(1).ecx & (1 << 12) != 0
}

/// Compare a rectangular destination region with nearest-neighbor samples.
///
/// The caller supplies contiguous byte masks, three bytes of trailing source
/// padding for the gathers, and ROI bounds inside the destination. The
/// inverse transform and pixel rounding follow OpenCV 5's AVX2 warp.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA. All pointers must be valid for the call.
/// `inverse` and `roi` refer to aligned arrays of six and four values.
#[no_mangle]
#[target_feature(enable = "avx2,fma")]
pub unsafe extern "C" fn affine_roi_mismatches(
    source: *const u8,
    source_width: i32,
    source_height: i32,
    target: *const u8,
    target_width: i32,
    inverse: *const f32,
    roi: *const i32,
) -> u32 {
    let [a, b, c, d, e, f] = inverse.cast::<[f32; 6]>().read();
    let [left, top, right, bottom] = roi.cast::<[i32; 4]>().read();
    let vector_end = target_width / 16 * 16;
    let vector_right = right.min(vector_end);
    let zero = _mm256_setzero_si256();
    let invalid_low = _mm256_set1_epi32(-1);
    let width = _mm256_set1_epi32(source_width);
    let height = _mm256_set1_epi32(source_height);
    let byte_mask = _mm256_set1_epi32(255);
    let column_scale_x = _mm256_set1_ps(a);
    let column_scale_y = _mm256_set1_ps(d);
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut total = 0;
    for y in top..bottom {
        let row_x = (y as f32) * b;
        let row_y = (y as f32) * e;
        let offset_x = _mm256_set1_ps(row_x + c);
        let offset_y = _mm256_set1_ps(row_y + f);
        let row = target.add((y as usize) * target_width as usize);
        let mut x = left;
        while x + 8 <= vector_right {
            let columns = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(x), lanes));
            let sx = round_coordinates(_mm256_fmadd_ps(columns, column_scale_x, offset_x));
            let sy = round_coordinates(_mm256_fmadd_ps(columns, column_scale_y, offset_y));
            let valid_x = _mm256_and_si256(
                _mm256_cmpgt_epi32(sx, invalid_low),
                _mm256_cmpgt_epi32(width, sx),
            );
            let valid_y = _mm256_and_si256(
                _mm256_cmpgt_epi32(sy, invalid_low),
                _mm256_cmpgt_epi32(height, sy),
            );
            let valid = _mm256_and_si256(valid_x, valid_y);
            let indices = _mm256_add_epi32(_mm256_mullo_epi32(sy, width), sx);
            let samples = _mm256_mask_i32gather_epi32::<1>(zero, source.cast(), indices, valid);
            let samples = _mm256_and_si256(samples, byte_mask);
            let expected = _mm256_cvtepu8_epi32(_mm_loadl_epi64(row.add(x as usize).cast()));
            let equal = _mm256_castsi256_ps(_mm256_cmpeq_epi32(samples, expected));
            total += 8 - (_mm256_movemask_ps(equal) as u32).count_ones();
            x += 8;
        }
        while x < right {
            let column = _mm_set_ss(x as f32);
            let (sx, sy) = if x < vector_end {
                (
                    _mm_fmadd_ss(column, _mm_set_ss(a), _mm_set_ss(row_x + c)),
                    _mm_fmadd_ss(column, _mm_set_ss(d), _mm_set_ss(row_y + f)),
                )
            } else {
                (
                    _mm_set_ss(((x as f32) * a + row_x) + c),
                    _mm_set_ss(((x as f32) * d + row_y) + f),
                )
            };
            let sx = round_coordinate(sx);
            let sy = round_coordinate(sy);
            let sample = if (sx as u32) < source_width as u32 && (sy as u32) < source_height as u32
            {
                *source.add((sy * source_width + sx) as usize)
            } else {
                0
            };
            total += u32::from(sample != *row.add(x as usize));
            x += 1;
        }
    }
    total
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn round_coordinates(values: __m256) -> __m256i {
    _mm256_cvttps_epi32(_mm256_round_ps::<
        { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
    >(values))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn round_coordinate(value: __m128) -> i32 {
    _mm_cvttss_si32(_mm_round_ss::<
        { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
    >(value, value))
}
