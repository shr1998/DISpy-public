"""Resolve dispersion parameters from configuration and CCF metadata."""

from __future__ import annotations

from typing import Any, Mapping


CC_DATA_PARAMETER_KEYS = (
    "cc_len",
    "dr",
    "step",
    "c_range",
    "samplerate",
    "samplerate_in_use",
    "delta",
    "dt",
)

CC_FILE_PARAMETER_ALIASES = {
    "channel_spacing_m": "dr",
    "source_step_channels": "step",
    "c_range_channels": "c_range",
    "samplerate_in_use_hz": "samplerate_in_use",
    "delta_s": "dt",
}


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _sampling_interval(params: Mapping[str, Any] | None) -> float | None:
    if not params:
        return None
    for key in ("dt", "delta"):
        value = _positive_number(params.get(key))
        if value is not None:
            return value
    for key in ("samplerate_in_use", "samplerate"):
        value = _positive_number(params.get(key))
        if value is not None:
            return 1.0 / value
    return None


def resolve_dispersion_params(
    dispersion_params: Mapping[str, Any] | None,
    cc_params: Mapping[str, Any] | None = None,
    cc_file_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge parameters with CCF-file metadata taking highest priority.

    Only acquisition and CCF geometry values are inherited. Dispersion choices
    such as frequency/velocity limits and ``time_range`` stay under the
    dispersion configuration.
    """

    resolved = dict(dispersion_params or {})
    canonical_file_params = dict(cc_file_params or {})
    for file_key, canonical_key in CC_FILE_PARAMETER_ALIASES.items():
        if file_key in canonical_file_params and canonical_key not in canonical_file_params:
            canonical_file_params[canonical_key] = canonical_file_params[file_key]

    for source in (cc_params or {}, canonical_file_params):
        for key in CC_DATA_PARAMETER_KEYS:
            if key in source and source[key] is not None:
                resolved[key] = source[key]

    sampling_interval = _sampling_interval(canonical_file_params)
    if sampling_interval is None:
        sampling_interval = _sampling_interval(cc_params)
    if sampling_interval is not None:
        resolved["dt"] = sampling_interval

    return resolved
