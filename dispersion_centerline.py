import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    distance_transform_edt,
    gaussian_filter1d,
    label,
    median_filter,
)


def extract_dispersion_centerlines(
    probability,
    extent,
    threshold=0.45,
    min_area=40,
    min_width=8,
    max_gap=4,
    smooth_sigma=1.5,
    sample_step=2,
):
    """Extract one frequency-velocity centerline from each connected energy band."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2:
        raise ValueError("probability must be a 2-D velocity-by-frequency array")

    finite = np.isfinite(probability)
    if not np.any(finite):
        return [], np.zeros_like(probability, dtype=np.uint8)

    probability = np.where(finite, probability, 0.0)
    peak = float(np.max(probability))
    if peak <= 0:
        return [], np.zeros_like(probability, dtype=np.uint8)
    if peak > 1.0:
        probability = probability / peak

    structure = np.ones((3, 3), dtype=bool)
    mask = probability >= float(threshold)
    mask = binary_closing(mask, structure=structure, iterations=1)
    mask = binary_opening(mask, structure=structure, iterations=1)
    components, count = label(mask, structure=structure)

    curves_px = []
    accepted_mask = np.zeros_like(mask, dtype=np.uint8)
    for component_id in range(1, count + 1):
        component = components == component_id
        rows, cols = np.nonzero(component)
        if rows.size < int(min_area) or np.unique(cols).size < int(min_width):
            continue

        distance = distance_transform_edt(component)
        x_values = np.arange(cols.min(), cols.max() + 1)
        centers = np.full(x_values.size, np.nan, dtype=float)
        for i, x in enumerate(x_values):
            y_values = np.flatnonzero(component[:, x])
            if y_values.size == 0:
                continue
            weights = distance[y_values, x] ** 2
            centers[i] = (
                np.average(y_values, weights=weights)
                if np.sum(weights) > 0
                else 0.5 * (y_values[0] + y_values[-1])
            )

        valid = np.isfinite(centers)
        if np.count_nonzero(valid) < int(min_width):
            continue

        for missing_index in np.flatnonzero(~valid):
            left = np.flatnonzero(valid[:missing_index])
            right = np.flatnonzero(valid[missing_index + 1:])
            if left.size == 0 or right.size == 0:
                continue
            left_index = int(left[-1])
            right_index = int(missing_index + 1 + right[0])
            if right_index - left_index - 1 <= int(max_gap):
                centers[missing_index] = np.interp(
                    missing_index,
                    [left_index, right_index],
                    [centers[left_index], centers[right_index]],
                )

        valid = np.isfinite(centers)
        x_values = x_values[valid]
        centers = centers[valid]
        if centers.size < int(min_width):
            continue

        centers = median_filter(centers, size=5, mode="nearest")
        centers = gaussian_filter1d(centers, sigma=float(smooth_sigma), mode="nearest")
        curves_px.append(np.column_stack((x_values, centers)))
        accepted_mask[component] = 1

    fmin, fmax, vmin, vmax = np.asarray(extent, dtype=float)
    height, width = probability.shape
    curves = []
    for curve_px in curves_px:
        curve_px = curve_px[::max(1, int(sample_step))]
        frequency = fmin + curve_px[:, 0] * (fmax - fmin) / max(width - 1, 1)
        velocity = vmin + curve_px[:, 1] * (vmax - vmin) / max(height - 1, 1)
        curves.append(np.column_stack((frequency, velocity)))

    curves.sort(key=lambda curve: float(np.nanmedian(curve[:, 1])))
    return curves, accepted_mask


def combine_centerlines(curves):
    valid = [np.asarray(curve, dtype=float) for curve in curves if len(curve)]
    if not valid:
        return np.empty((0, 2), dtype=float)
    return np.vstack(valid)


def order_centerlines_by_min_frequency(curves):
    """Return curve indices ordered by their finite lower-frequency limit."""
    ranked = []
    for index, curve in enumerate(curves):
        array = np.asarray(curve, dtype=float)
        if array.ndim == 2 and array.shape[1] and len(array):
            finite_frequency = array[:, 0][np.isfinite(array[:, 0])]
        else:
            finite_frequency = np.empty(0, dtype=float)
        lower_frequency = (
            float(np.min(finite_frequency)) if len(finite_frequency) else float("inf")
        )
        ranked.append((lower_frequency, index))
    return [index for _, index in sorted(ranked)]
