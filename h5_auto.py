# -*- coding: utf-8 -*-
"""Helpers for robust HDF5 data and parameter discovery.

The project historically assumed Silixa-like paths such as
``Acquisition/Raw[0]/RawData``.  These helpers keep that format working while
also scanning arbitrary H5 files for equivalent datasets and attributes.
"""

from __future__ import annotations

import re
import os
import datetime as _datetime
from typing import Any

import h5py
import numpy as np

try:
    import obspy
except Exception:  # pragma: no cover - obspy is available in the app env.
    obspy = None


DATASET_NAME_HINTS = (
    "rawdata",
    "raw_data",
    "data",
    "strain",
    "das",
    "trace",
    "traces",
)

PARAM_KEYWORDS = {
    "samplerate": (
        "outputdatarate",
        "samplerate",
        "sample_rate",
        "samplingrate",
        "sampling_rate",
        "samplingfrequency",
        "sampling_frequency",
        "fs",
    ),
    "dr": (
        "spatialsamplinginterval",
        "spatial_sampling_interval",
        "channelspacing",
        "channel_spacing",
        "receiverinterval",
        "receiver_interval",
        "dx",
        "dr",
    ),
    "starttime": (
        "partstarttime",
        "starttime",
        "start_time",
        "begintime",
        "begin_time",
        "gpstimestamp",
    ),
    "endtime": (
        "partendtime",
        "endtime",
        "end_time",
        "stoptime",
        "stop_time",
    ),
}


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if isinstance(value, np.bytes_):
        return bytes(value).decode(errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode_value(value.item())
        return [_decode_value(v) for v in value.tolist()]
    return value


def _as_float(value: Any) -> float | None:
    value = _decode_value(value)
    try:
        if isinstance(value, str):
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
            if not match:
                return None
            return float(match.group(0))
        return float(value)
    except Exception:
        return None


def _as_time(value: Any) -> Any:
    value = _decode_value(value)
    if obspy is None:
        if isinstance(value, str):
            text = value.strip().replace("Z", "")
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                        "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
                try:
                    return _datetime.datetime.strptime(text, fmt)
                except Exception:
                    pass
        return value
    try:
        return obspy.UTCDateTime(value)
    except Exception:
        return value


def scan_h5(file_or_path):
    """Return dataset and attribute metadata for an open file or path."""
    close_after = False
    if isinstance(file_or_path, (str, bytes, os.PathLike)):
        h5 = h5py.File(file_or_path, "r")
        close_after = True
    else:
        h5 = file_or_path

    datasets = {}
    attrs = {}
    try:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[obj.name.lstrip("/")] = {
                    "shape": tuple(obj.shape),
                    "ndim": int(obj.ndim),
                    "dtype": str(obj.dtype),
                    "size": int(np.prod(obj.shape)) if obj.shape else 1,
                }
            if hasattr(obj, "attrs"):
                for attr_name, attr_value in obj.attrs.items():
                    attrs[f"{obj.name.lstrip('/')}@{attr_name}"] = _decode_value(attr_value)

        h5.visititems(visitor)
        for attr_name, attr_value in h5.attrs.items():
            attrs[f"/@{attr_name}"] = _decode_value(attr_value)
    finally:
        if close_after:
            h5.close()
    return datasets, attrs


def find_data_struct(file_or_path, preferred=None):
    """Find the most likely numeric signal dataset path in an H5 file."""
    close_after = False
    if isinstance(file_or_path, (str, bytes, os.PathLike)):
        h5 = h5py.File(file_or_path, "r")
        close_after = True
    else:
        h5 = file_or_path

    try:
        if preferred and preferred in h5 and isinstance(h5[preferred], h5py.Dataset):
            return preferred

        classic = "Acquisition/Raw[0]/RawData"
        if classic in h5 and isinstance(h5[classic], h5py.Dataset):
            return classic

        candidates = []

        def visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.ndim < 2:
                return
            if not np.issubdtype(obj.dtype, np.number):
                return
            clean_name = name.lower().replace("/", "")
            hint_score = 0
            for hint in DATASET_NAME_HINTS:
                if hint in clean_name:
                    hint_score += 10
            candidates.append((hint_score, int(np.prod(obj.shape)), name))

        h5.visititems(visitor)
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]
    finally:
        if close_after:
            h5.close()


def _find_attr(attrs, keywords):
    for full_key, value in attrs.items():
        key = full_key.lower().replace(" ", "").replace("-", "_")
        key_compact = key.replace("_", "")
        for keyword in keywords:
            keyword = keyword.lower()
            if keyword in key or keyword.replace("_", "") in key_compact:
                return value, full_key
    return None, None


def read_h5_params(file_or_path, data_struct=None):
    """Auto-read common acquisition parameters from H5 attributes."""
    datasets, attrs = scan_h5(file_or_path)
    params = {}

    if data_struct is None:
        data_struct = find_data_struct(file_or_path)
    if data_struct is not None:
        params["data_struct"] = str(data_struct).lstrip("/")

    samplerate, samplerate_key = _find_attr(attrs, PARAM_KEYWORDS["samplerate"])
    dr, dr_key = _find_attr(attrs, PARAM_KEYWORDS["dr"])
    starttime, start_key = _find_attr(attrs, PARAM_KEYWORDS["starttime"])
    endtime, end_key = _find_attr(attrs, PARAM_KEYWORDS["endtime"])

    samplerate_float = _as_float(samplerate)
    if samplerate_float and samplerate_float > 0:
        params["samplerate"] = samplerate_float
        params["delta"] = 1.0 / samplerate_float
        params["samplerate_source"] = samplerate_key

    dr_float = _as_float(dr)
    if dr_float is not None:
        params["dr"] = dr_float
        params["dr_source"] = dr_key

    if starttime is not None:
        params["starttime"] = _as_time(starttime)
        params["starttime_source"] = start_key
    if endtime is not None:
        params["endtime"] = _as_time(endtime)
        params["endtime_source"] = end_key

    if "starttime" in params and "endtime" in params:
        try:
            time_len = params["endtime"] - params["starttime"]
            if hasattr(time_len, "total_seconds"):
                time_len = time_len.total_seconds()
            params["time_len"] = time_len
        except Exception:
            pass

    return params


def read_h5_data(cur_file, data_struct=None, transpose=True):
    """Read signal data using an explicit or auto-detected dataset path."""
    with h5py.File(cur_file, "r") as h5:
        data_struct = find_data_struct(h5, preferred=data_struct)
        if data_struct is None:
            raise KeyError(f"No suitable numeric 2D dataset found in {cur_file}")
        data = np.asarray(h5[data_struct][:])
    if transpose and data.ndim == 2:
        data = data.transpose()
    return data, data_struct


def read_data_compat(cur_file, data_struct=None, para=False, data_temp=True, transpose=True):
    """Compatibility wrapper for existing read_data signatures."""
    data = None
    resolved_struct = data_struct
    if data_temp:
        data, resolved_struct = read_h5_data(cur_file, data_struct=data_struct, transpose=transpose)
    elif data_struct is None:
        resolved_struct = find_data_struct(cur_file)

    if para:
        params = read_h5_params(cur_file, data_struct=resolved_struct)
        if data_temp:
            return data, params
        return params
    return data
