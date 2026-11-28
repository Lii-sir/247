/// Select the first depth sample nearest the linear percentile of each segment.
/// Scratch storage is reused across segments. Empty segments produce -1.
pub fn select_samples<T: Copy + Ord>(
    values: &[T],
    counts: &[i64],
    percentile: f64,
    scratch: &mut [T],
    indices: &mut [i64],
    as_float: impl Fn(T) -> f64,
) -> bool {
    if !(0.0..=100.0).contains(&percentile) || counts.len() != indices.len() {
        return false;
    }

    let quantile = percentile / 100.0;
    let mut start = 0usize;
    for (&count, result) in counts.iter().zip(indices) {
        let Ok(count) = usize::try_from(count) else {
            return false;
        };
        if count == 0 {
            *result = -1;
            continue;
        }
        let Some(end) = start.checked_add(count) else {
            return false;
        };
        let (Some(samples), Some(work)) = (values.get(start..end), scratch.get_mut(..count)) else {
            return false;
        };
        work.copy_from_slice(samples);

        let rank = (count - 1) as f64 * quantile;
        let lower = rank as usize;
        let fraction = rank - lower as f64;
        let (_, a, above) = work.select_nth_unstable(lower);
        let a = as_float(*a);
        let target = if fraction == 0.0 {
            a
        } else {
            let Some(&upper) = above.iter().min() else {
                return false;
            };
            let b = as_float(upper);
            let delta = b - a;
            if fraction < 0.5 {
                a + delta * fraction
            } else {
                b - delta * (1.0 - fraction)
            }
        };

        let mut nearest = f64::INFINITY;
        let mut index = start;
        for (offset, &sample) in samples.iter().enumerate() {
            let distance = (as_float(sample) - target).abs();
            if distance < nearest {
                nearest = distance;
                index = start + offset;
                if distance == 0.0 {
                    break;
                }
            }
        }
        *result = index as i64;
        start = end;
    }
    start == values.len()
}

#[cfg(test)]
mod tests {
    use super::select_samples;

    #[test]
    fn keeps_original_order_on_ties_and_reports_empty_segments() {
        let mut scratch = [0u16; 3];
        let mut indices = [0; 4];
        assert!(select_samples(
            &[8u16, 4, 2, 12, 10],
            &[0, 2, 0, 3],
            50.0,
            &mut scratch,
            &mut indices,
            f64::from,
        ));
        assert_eq!(indices, [-1, 0, -1, 4]);
    }

    #[test]
    fn rejects_invalid_counts_and_percentile() {
        let mut scratch = [0u16; 2];
        let mut indices = [0];
        for (count, percentile) in [(-1, 95.0), (1, 95.0), (3, 95.0), (2, f64::NAN)] {
            assert!(!select_samples(
                &[3u16, 4],
                &[count],
                percentile,
                &mut scratch,
                &mut indices,
                f64::from,
            ));
        }
    }
}
