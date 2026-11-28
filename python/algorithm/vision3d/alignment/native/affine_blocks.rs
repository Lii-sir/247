//! Exact binary-mask scoring with a summed-area table of uniform source tiles.
use super::{mismatch8, pixel_loss, round_coordinate, round_coordinates};
use core::arch::x86_64::*;

const BLOCK_WIDTH: i32 = 64;

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn coordinate(x: i32, slope: __m128, bias: __m128) -> i32 {
    round_coordinate(_mm_fmadd_ss(_mm_set_ss(x as f32), slope, bias))
}

const TILE_X: usize = 8;
const TILE_Y: usize = 4;

/// # Safety
/// The caller provides both masks and room for (ceil(sw/8)+1) by
/// (ceil(sh/4)+1) packed prefixes and (tw+1) by th target prefixes.
#[no_mangle]
pub unsafe extern "C" fn build_affine_blocks(
    source: *const u8,
    sw: usize,
    sh: usize,
    target: *const u8,
    tw: usize,
    th: usize,
    tiles: *mut u64,
    prefix: *mut u16,
) -> bool {
    let cols = sw.div_ceil(TILE_X);
    let rows = sh.div_ceil(TILE_Y);
    let stride = cols + 1;
    for x in 0..stride {
        tiles.add(x).write_volatile(0);
    }
    for y in 0..rows {
        let row = tiles.add((y + 1) * stride);
        let previous = tiles.add(y * stride);
        row.write(0);
        let mut accum = 0u64;
        for x in 0..cols {
            let left = x * TILE_X;
            let top = y * TILE_Y;
            let right = (left + TILE_X).min(sw);
            let bottom = (top + TILE_Y).min(sh);
            let mut ones = 0usize;
            for sy in top..bottom {
                for sx in left..right {
                    let value = *source.add(sy * sw + sx);
                    if value > 1 {
                        return false;
                    }
                    ones += value as usize;
                }
            }
            let count = (right - left) * (bottom - top);
            accum += (ones > 0) as u64 + (((ones == count) as u64) << 32);
            row.add(x + 1).write(accum + *previous.add(x + 1));
        }
    }
    for y in 0..th {
        let row = prefix.add(y * (tw + 1));
        row.write(0);
        let mut count = 0u16;
        for x in 0..tw {
            let value = *target.add(y * tw + x);
            if value > 1 {
                return false;
            }
            count += value as u16;
            row.add(x + 1).write(count);
        }
    }
    true
}

#[inline]
unsafe fn classify_block(
    tiles: *const u64,
    sw: i32,
    sh: i32,
    stride: usize,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
) -> Option<u8> {
    let left = x0.min(x1);
    let right = x0.max(x1);
    let top = y0.min(y1);
    let bottom = y0.max(y1);
    if right < 0 || bottom < 0 || left >= sw || top >= sh {
        return Some(0);
    }
    let l = left.max(0) as usize / TILE_X;
    let r = right.min(sw - 1) as usize / TILE_X + 1;
    let t = top.max(0) as usize / TILE_Y;
    let b = bottom.min(sh - 1) as usize / TILE_Y + 1;
    let cells = (*tiles.add(b * stride + r))
        .wrapping_add(*tiles.add(t * stride + l))
        .wrapping_sub(*tiles.add(t * stride + r))
        .wrapping_sub(*tiles.add(b * stride + l));
    if cells as u32 == 0 {
        Some(0)
    } else if (cells >> 32) as usize == (r - l) * (b - t)
        && left >= 0
        && top >= 0
        && right < sw
        && bottom < sh
    {
        Some(1)
    } else {
        None
    }
}

/// # Safety
/// CPU supports AVX2/FMA and dimensions are below 32767; projected coordinates fit i32.
/// All ROI and output arrays have the declared lengths and rectangles lie inside target.
/// Source includes three gather padding bytes. Tables are from build_affine_blocks.
#[no_mangle]
#[target_feature(enable = "avx2,fma")]
pub unsafe extern "C" fn score_affine_blocks(
    source: *const u8,
    sw: i32,
    sh: i32,
    target: *const u8,
    tw: i32,
    tiles: *const u64,
    prefix: *const u16,
    inverse: *const f32,
    rois: *const i32,
    count: usize,
    output: *mut u32,
) {
    let [a, b, c, d, e, f] = inverse.cast::<[f32; 6]>().read();
    let stride = (sw as usize).div_ceil(TILE_X) + 1;
    let ax = _mm_set_ss(a);
    let ay = _mm_set_ss(d);
    let vx = _mm256_set1_ps(a);
    let vy = _mm256_set1_ps(d);
    let width = _mm256_set1_epi32(sw);
    let height = _mm256_set1_epi32(sh);
    let lanes = _mm256_setr_ps(0., 1., 2., 3., 4., 5., 6., 7.);
    let step = _mm256_set1_ps(8.);
    let vector_end = tw / 16 * 16;
    for index in 0..count {
        let [left, top, right, bottom] = rois.add(index * 4).cast::<[i32; 4]>().read();
        let vector_right = right.min(vector_end);
        let mut total = 0u32;
        for y in top..bottom {
            let px = prefix.add(y as usize * (tw as usize + 1));
            let row = target.add((y * tw) as usize);
            let row_x = y as f32 * b;
            let row_y = y as f32 * e;
            let bx = _mm_set_ss(row_x + c);
            let by = _mm_set_ss(row_y + f);
            let vbx = _mm256_set1_ps(row_x + c);
            let vby = _mm256_set1_ps(row_y + f);
            let mut x = left;
            while x < vector_right {
                let end = (x + BLOCK_WIDTH).min(vector_right);
                // Inverse coordinates on one destination row are monotone.
                // Their endpoints bound all source pixels sampled by this block.
                let sx0 = coordinate(x, ax, bx);
                let sx1 = coordinate(end - 1, ax, bx);
                let sy0 = coordinate(x, ay, by);
                let sy1 = coordinate(end - 1, ay, by);
                if let Some(value) = classify_block(tiles, sw, sh, stride, sx0, sx1, sy0, sy1) {
                    let ones = (*px.add(end as usize) - *px.add(x as usize)) as u32;
                    total += if value == 0 {
                        ones
                    } else {
                        (end - x) as u32 - ones
                    };
                    x = end;
                    continue;
                }
                let mut cols = _mm256_add_ps(_mm256_set1_ps(x as f32), lanes);
                while x + 8 <= end {
                    let sx = round_coordinates(_mm256_fmadd_ps(cols, vx, vbx));
                    let sy = round_coordinates(_mm256_fmadd_ps(cols, vy, vby));
                    total += mismatch8(source, row.add(x as usize), sx, sy, width, height);
                    cols = _mm256_add_ps(cols, step);
                    x += 8;
                }
                while x < end {
                    total += pixel_loss(
                        source,
                        sw,
                        sh,
                        coordinate(x, ax, bx),
                        coordinate(x, ay, by),
                        *row.add(x as usize),
                    );
                    x += 1;
                }
            }
            while x < right {
                let sx = round_coordinate(_mm_set_ss(((x as f32) * a + row_x) + c));
                let sy = round_coordinate(_mm_set_ss(((x as f32) * d + row_y) + f));
                total += pixel_loss(source, sw, sh, sx, sy, *row.add(x as usize));
                x += 1;
            }
        }
        output.add(index).write(total);
    }
}
