"""Select independent height-percentile sample indices from packed depth values."""
import numpy as np


def pick_height_samples(values: list[int], counts: list[int], percentile: float) -> list[int]:
    """Return the first nearest sample in each segment, or -1 for empty segments."""
    raw = np.asarray(values)
    sizes = np.asarray(counts, dtype=np.intp)
    if not 0 <= percentile <= 100 or (sizes < 0).any() or sizes.sum() != raw.size:
        return []
    chosen = np.full(len(sizes), -1, dtype=np.intp)
    active = np.flatnonzero(sizes)
    if not active.size:
        return chosen.tolist()

    rows = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.cumsum(sizes)[active] - sizes[active]
    ordered = raw[np.lexsort((raw, rows))]
    rank = (sizes[active] - 1) * (percentile / 100.0)
    low = rank.astype(np.intp)
    fraction = rank - low
    a = ordered[starts + low].astype(np.float64)
    b = ordered[starts + np.minimum(low + 1, sizes[active] - 1)].astype(np.float64)
    delta = b - a
    percentiles = np.empty(len(sizes), dtype=np.float64)
    # Match NumPy's linear percentile interpolation on both sides of the midpoint.
    percentiles[active] = np.where(fraction < .5, a + delta * fraction, b - delta * (1.0 - fraction))
    distance = np.abs(raw - percentiles[rows])
    nearest = np.empty(len(sizes))
    nearest[active] = np.minimum.reduceat(distance, starts)
    matches = np.flatnonzero(distance == nearest[rows])
    matched_rows = rows[matches]
    chosen[active] = matches[np.r_[True, matched_rows[1:] != matched_rows[:-1]]]
    return chosen.tolist()
