# -*- coding: utf-8 -*-
# @Time : 2024/11/26 13:30
# @Site :
# @File : utillis_new.py
# @Software: PyCharm

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import datetime

import torch
from cupy.fft import rfftfreq, rfft
from matplotlib import pyplot as plt
try:
    from nptdms import TdmsFile
except ImportError:
    TdmsFile = None
from numba import njit, prange
import numba as nb
import numpy as np
import h5py
from obspy import Trace, Stream
from obspy.geodetics import gps2dist_azimuth
from obspy.io.sac import SACTrace
try:
    from pyproj import Transformer
except ImportError:
    Transformer = None
from scipy import signal
from scipy.fftpack import next_fast_len
import glob
import cupyx.scipy.ndimage as nd
import obspy
from tqdm import tqdm
from scipy.signal import convolve2d, windows
import pynvml
import cupy as cp
from dataclasses import dataclass
from h5_auto import find_data_struct, read_data_compat


@dataclass
class BeamformResult:
    det_azs: np.ndarray
    all_beam_power: list
    slowness: np.ndarray
    azimuths: np.ndarray
    slowness_indx: np.ndarray


def compute_fx_spectrum(data, fs, window_length, overlap=0.5, nfft=None):
    """
    计算多道地震数据的长时间叠加F-X谱
    
    参数：
        data : 二维numpy数组，形状为（n_traces, n_samples）
               - 行表示地震道
               - 列表示时间采样点
        fs : 采样率（Hz）
        window_length : 时间窗长度（秒）
        overlap : 时间窗重叠比例（0到1之间，默认0.5）
        nfft : FFT点数（默认None表示使用窗口长度）
        
    返回：
        fx_spectrum : 二维复数数组，形状为（n_freqs, n_traces）
                       - 频率-空间域的复频谱
        freqs : 一维数组，频率坐标（Hz）
    """
    # 参数校验
    n_traces, n_samples = data.shape
    if window_length <= 0:
        raise ValueError("Window length must be greater than zero")
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in the range [0, 1)")

    # 计算窗口样本数
    win_samples = int(window_length * fs)
    if win_samples > n_samples:
        raise ValueError("Window length exceeds the total data duration")

    # 设置FFT点数
    nfft = nfft or win_samples

    # 计算跳点数
    hop_samples = int(win_samples * (1 - overlap))

    # 创建汉宁窗
    window = signal.windows.hann(win_samples)

    # 初始化频谱累加器
    sum_spectrum = np.zeros((nfft // 2 + 1, n_traces), dtype=np.complex128)
    frame_count = 0

    # 时间窗滑动处理
    for start_idx in range(0, n_samples - win_samples + 1, hop_samples):
        end_idx = start_idx + win_samples

        # 提取当前时间窗数据
        windowed_data = data[:, start_idx:end_idx] * window

        # 计算各道FFT
        fft_result = np.fft.rfft(windowed_data, n=nfft, axis=1)

        # 频谱累加
        sum_spectrum += fft_result.T  # 转置为（freqs, traces）
        frame_count += 1

    # 计算平均频谱
    avg_spectrum = sum_spectrum / frame_count

    # 生成频率轴
    freqs = np.fft.rfftfreq(nfft, 1 / fs)

    return avg_spectrum, freqs


def gaussian_filter_scipy(array, sigma=1, kernel_size=3):
    # 生成一维高斯核
    gauss_1d = windows.gaussian(kernel_size, sigma)
    # 生成二维高斯核（外积）
    kernel = np.outer(gauss_1d, gauss_1d)
    # 归一化核
    kernel /= np.sum(kernel)
    # 执行卷积（边界填充方式可选：'same', 'valid'等）
    smoothed = convolve2d(array, kernel, mode='same', boundary='symm')
    return smoothed


def fk_f(data_filter, dt, dx, w=20, show=False, freq_window=5):
    d = data_filter
    n = len(d[0])
    # m为空间方向的采样点数，m增大可以让FK谱光滑一点，以达到插值效果。
    m = len(d[:, 0])
    D = np.zeros((m, n))
    D[:len(d[:, 0])] = d
    # 时间采样率。
    fs = 1 / dt
    # 空间采样率
    xs = 1 / dx
    # 频率 (赫兹)。
    f = np.arange(-n // 2, n // 2) * fs / (n - 1)
    # 波数 (每米)。
    k = 2 * np.pi * np.arange(-m // 2, m // 2) * xs / (m - 1)
    # 二维FFT。
    fk = np.fft.fft2(D)
    if w:
        # N = 9
        # K = 1  # 45度对角线
        # filter_window = fill_slope_matrix_precise(N, K)
        mask = np.ones(fk.shape)
        h = len(mask)
        l = len(mask[1])
        temp = (w * h / 2) / l
        for i in range(len(mask[0])):
            # mask[max(int(h / 2), int(round(h / 2 - i * temp))):max(int(h / 2 + 1), int(round(h / 2 + i * temp))), i] = 0
            if i > len(mask[0]) / 2:
                mask[0:max(0, int(round((l - i) * temp))), l - i - 1] = 0
                mask[max(0, int(round(h - (l - i) * temp))):h, l - i - 1] = 0
            else:
                mask[0:max(0, int(round(abs((l - i) * temp - temp * l)))), l - i - 1] = 0
                mask[max(0, int(round(h - abs((l - i) * temp - temp * l)))):h, l - i - 1] = 0
        smoothed_array = gaussian_filter_scipy(mask, sigma=6, kernel_size=6)
        fk = fk * smoothed_array
    if show:
        freq_window = freq_window / dt
        fk_1 = fk.copy()
        pmin = -10
        fs = 1 / dt
        lf = len(f)
        P = abs(np.fft.fftshift(fk_1))
        # N = 9
        # K = 0.1  # 45度对角线
        # filter_window = fill_slope_matrix_precise(N, K)
        # P = custom_median_filter(P, filter_window)
        P = P / P.max()
        # self.P = 10 * np.log10(P)
        P2 = abs(fk)
        P2 /= P2.max()
        # P2 = 10 * np.log10(P2)

    PP = np.fft.ifft2(fk)
    return PP.real


@nb.njit(nogil=True, fastmath=True, parallel=True, cache=True)
def numba_process(data_all, all_stdS):
    """
    Numba 加速版本，demean/detrend/taper
    输入要求:
        data_all: 形状为 [num_sta, num_seg, npts] 的连续内存数组
        all_stdS: 形状为 [num_sta] 的每个 station 的标准差
    """
    num_sta, num_seg, npts = data_all.shape
    dataS = np.empty_like(data_all)
    trace_stdS = np.empty((num_sta, num_seg))

    # 预生成汉宁窗 (与 np.hanning(npts) 一致)
    window = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(npts) / max(1, npts - 1)))

    # 并行处理每个 segment
    for iseg in nb.prange(num_seg):
        for ist in range(num_sta):
            segment = data_all[ist, iseg, :].copy()

            # ---- 1. 去均值 ----
            mean = np.mean(segment)
            segment -= mean

            # ---- 2. 去趋势 (完全复现 scipy.signal.detrend) ----
            n = len(segment)
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_x2 = 0.0

            for i in range(n):
                x = float(i)
                y = segment[i]
                sum_x += x
                sum_y += y
                sum_xy += x * y
                sum_x2 += x ** 2

            denominator = n * sum_x2 - sum_x ** 2
            if denominator == 0:
                slope = 0.0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

            for i in range(n):
                segment[i] -= slope * i + intercept

            # ---- 3. 加窗 ----
            for i in range(n):
                segment[i] *= window[i]

            # 存储结果
            dataS[ist, iseg, :] = segment

            # 计算 trace_stdS
            max_abs = np.max(np.abs(segment))
            trace_stdS[ist, iseg] = max_abs / all_stdS

    return dataS, trace_stdS


@nb.njit(nogil=True, fastmath=True, parallel=True, cache=True)
def numba_process1(data_all, all_stdS):
    num_sta, num_seg, npts = data_all.shape
    dataS = np.empty_like(data_all)
    trace_stdS = np.empty((num_sta, num_seg))

    # 汉宁窗生成 (修复npts=1时的分母问题)
    window = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(npts) / (npts if npts <= 1 else npts - 1)))

    for iseg in nb.prange(num_seg):
        for ist in range(num_sta):
            segment = data_all[ist, iseg, :].copy()

            # ---- 1. 去均值 ----
            mean = np.mean(segment)
            segment -= mean

            # ---- 2. 去趋势 (优化数值稳定性) ----
            n = len(segment)
            sum_x = sum_y = sum_xy = sum_x2 = 0.0
            for i in range(n):
                x = i  # 无需强制转float
                y = segment[i]
                sum_x += x
                sum_y += y
                sum_xy += x * y
                sum_x2 += x * x

            denominator = n * sum_x2 - sum_x ** 2
            slope = (n * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0.0
            intercept = (sum_y - slope * sum_x) / n

            for i in range(n):
                segment[i] -= slope * i + intercept

            # ---- 3. 加窗 (向量化操作) ----
            segment *= window  # 直接数组相乘

            dataS[ist, iseg, :] = segment

            max_abs = np.max(np.abs(segment))  # 移除axis参数
            trace_stdS[ist, iseg] = max_abs / all_stdS[ist]  # 按station索引

    return dataS, trace_stdS


def chunked_numba_process(data_all, all_stdS, chunk_size=500):
    """分块处理避免内存不足"""
    num_seg = data_all.shape[1]
    dataS = np.empty_like(data_all)
    trace_stdS = np.empty((data_all.shape[0], num_seg))

    for start in range(0, num_seg, chunk_size):
        end = min(start + chunk_size, num_seg)
        chunk = data_all[:, start:end, :]
        dataS_chunk, trace_chunk = numba_process(chunk, all_stdS, chunk.shape[2])
        dataS[:, start:end, :] = dataS_chunk
        trace_stdS[:, start:end] = trace_chunk

    return dataS, trace_stdS


@njit(parallel=True)
def allstd_global_3d(data_all):
    total_elements = data_all.size  # 获取所有元素的总数
    sum_val = 0.0
    sum_sq_val = 0.0
    # 并行遍历所有元素
    for i in prange(total_elements):
        val = data_all.ravel()[i]  # ravel() 将 3D 数组展平成 1D，逐个访问元素
        sum_val += val
        sum_sq_val += val ** 2
    mean_val = sum_val / total_elements  # 计算均值
    mean_sq_val = sum_sq_val / total_elements  # 计算平方均值
    return np.sqrt(mean_sq_val - mean_val ** 2)  # 标准差公式


@njit(parallel=True)
def allstd_global(data_all):
    K, M, N = data_all.shape
    std_devs = np.zeros(K)  # 存储每个切片的计算结果

    # 并行遍历每个第一维切片
    for k in prange(K):
        # 提取当前切片并展平为1D数组
        slice_flat = data_all[k].ravel()
        total = M * N

        # 计算当前切片的累加和与平方和
        sum_val = 0.0
        sum_sq_val = 0.0
        for i in range(total):
            val = slice_flat[i]
            sum_val += val
            sum_sq_val += val ** 2

        # 计算均值和标准差
        mean = sum_val / total
        std = np.sqrt((sum_sq_val / total) - (mean ** 2))
        std_devs[k] = std

    return std_devs


def get_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free / (1024 ** 2)


def read_data(cur_file, data_struct=None, para=False, data_temp=True):
    return read_data_compat(cur_file, data_struct=data_struct, para=para, data_temp=data_temp)


def read_data_tdms(fc_para, source):
    trace_stdS = []  # np.zeros((station_num, nseg), dtype=np.float32)
    dataS = []  # np.zeros(shape=(station_num, nseg, npts), dtype=np.float32)
    dataS_t = []  # np.zeros((station_num, nseg), dtype=np.float32)
    buffer = []
    trace_stdS = []
    starttime_buffer = False
    header = source['HEADER']
    data_all = source['data']
    station_num = data_all.shape[0]
    if data_all.ndim == 1:
        arr = data_all[np.newaxis, :]

    # load parameter from dic
    inc_mins = fc_para['inc_mins']
    cc_len = fc_para['cc_len']
    step = fc_para['step']
    demidean = fc_para['demidean']

    # useful parameters for trace sliding
    nseg = int(np.floor((inc_mins * 60 - cc_len) / step)) + 1
    sps = int(header['SamplingFrequency[Hz]'])

    if not starttime_buffer:
        starttime = datetime.datetime.strptime(header['GPSTimeStamp'], "%Y-%m-%dT%H:%M:%S.%f").timestamp()

    # all_stdS = np.std(data_all, axis=-1)  # standard deviation over all noise window

    # initialize variables

    window_size = cc_len * sps

    indx1 = 0

    if buffer != []:
        data_all = np.concatenate((buffer, data_all), axis=1)

    npts = data_all.shape[1]

    i = 0
    # if demidean == 'yes':
    # data_temp = cp.array(data_all)
    # data_all = median_filter(data_temp, size=(1, 40)).get()

    all_stdS = np.std(data_all, axis=-1)

    while i + window_size <= npts:
        window = data_all[:, i:i + window_size].copy()
        trace_stdS.append(np.max(np.abs(window), axis=-1) / all_stdS)

        # if demidean == 'yes':
        #     window = cp.array(window)
        #     window = median_filter(window, size=(1, 10)).get()

        # if demidean == 'yes':
        #     window = median_filter(window, size=(1, 150))
        # window = fk_filt(window, sps, header['SpatialResolution[m]'], sgn='neg', cmin=20,
        #        cmax=250)

        window = detrend(window)
        window = demean(window)
        window = taper(window)

        dataS.append(window)
        starttime = starttime + step
        starttime_buffer = True
        dataS_t.append(starttime)
        # if len(dataS) == max_nseg:
        #     #write_segment_data_h5(props, parameters, trace_stdS, dataS_t, dataS)
        #     write_segment_data_h5(props, parameters, np.array(trace_stdS).T, np.array(dataS_t), np.transpose(np.array(dataS), (1, 0, 2)))
        #     print('write out {} segments'.format(len(dataS)))
        #     print(np.array(trace_stdS).shape)
        #     # save
        #     dataS = []
        #     dataS_t = []
        #     trace_stdS = []
        i += step * sps


def get_file_list(base_files):
    """预处理获取所有需要处理的文件路径"""
    all_files = []
    for base_path in base_files:
        if os.path.isfile(base_path) and base_path.lower().endswith((".h5", ".hdf5")):
            all_files.append(base_path)
        elif os.path.isdir(base_path):
            files = sorted(glob.glob(os.path.join(base_path, "*.h5")))
            files.extend(sorted(glob.glob(os.path.join(base_path, "*.hdf5"))))
            all_files.extend(files)
    return all_files


def get_file_list_tdms(base_files):
    """预处理获取所有需要处理的文件路径"""
    all_files = []
    for base_path in base_files:
        if os.path.isfile(base_path) and base_path.lower().endswith(".tdms"):
            all_files.append(base_path)
        elif os.path.isdir(base_path):
            all_files.extend(sorted(glob.glob(os.path.join(base_path, "*.tdms"))))
    return all_files


def demean(arr): return arr - np.mean(arr, axis=-1, keepdims=True)


def split_dispersion_modes_2xn(
    disp,
    min_points_per_mode=3,
    tol=30,   # 允许的相速度抖动
    max_modes=None,
):
    disp = np.asarray(disp)
    assert disp.ndim == 2 and disp.shape[0] == 2

    vel = np.asarray(disp[0], dtype=float)
    freq = np.asarray(disp[1], dtype=float)

    valid = np.isfinite(vel) & np.isfinite(freq)
    vel = vel[valid]
    freq = freq[valid]
    if len(vel) == 0:
        return {}

    def _single_mode_result(v_arr, f_arr):
        if len(v_arr) == 0:
            return {}
        order_i = np.argsort(f_arr)
        pts = np.vstack((v_arr[order_i], f_arr[order_i]))
        return {0: pts}

    rounded_freq = np.round(freq, decimals=6)
    uniq_freq, uniq_idx = np.unique(rounded_freq, return_index=True)
    _, freq_counts = np.unique(rounded_freq, return_counts=True)
    sparse_point_count = len(vel)
    if sparse_point_count <= max(6, 2 * int(min_points_per_mode)):
        return _single_mode_result(vel[uniq_idx], freq[uniq_idx])

    if freq_counts.size > 0:
        multi_pick_ratio = float(np.mean(freq_counts > 1))
        if int(np.max(freq_counts)) <= 1 or multi_pick_ratio <= 0.12:
            return _single_mode_result(vel[uniq_idx], freq[uniq_idx])

    order = np.lexsort((vel, freq))
    vel = vel[order]
    freq = freq[order]

    rounded_freq = np.round(freq, decimals=6)
    uniq_freq = np.unique(rounded_freq)
    freq_spacing = np.diff(uniq_freq)
    base_df = float(np.median(np.abs(freq_spacing))) if freq_spacing.size > 0 else 1.0
    max_skip_steps = 1

    def _branch_predict(branch, f_now):
        pts = branch["points"]
        if len(pts) < 2:
            return float(branch["last_v"])
        v1, f1 = pts[-2]
        v2, f2 = pts[-1]
        df = float(f2 - f1)
        if abs(df) < 1.0e-8:
            return float(v2)
        slope = float((v2 - v1) / df)
        return float(v2 + slope * (f_now - f2))

    def _branch_scale(branch):
        pts = branch["points"]
        if len(pts) < 2:
            return 0.0
        vals = np.asarray([p[0] for p in pts[-4:]], dtype=float)
        if vals.size < 2:
            return 0.0
        return float(np.median(np.abs(np.diff(vals))))

    branches = []
    for f in uniq_freq:
        idx = np.where(rounded_freq == f)[0]
        cand_vel = vel[idx]
        cand_pairs = sorted(zip(cand_vel, idx), key=lambda x: x[0])

        for branch in branches:
            branch["matched"] = False

        for v, src_idx in cand_pairs:
            best_branch = None
            best_score = None
            for branch in branches:
                if branch["matched"] or branch.get("closed", False):
                    continue

                missed = int(branch.get("missed", 0))
                if missed > max_skip_steps:
                    continue

                pred_v = _branch_predict(branch, f)
                last_gap = abs(float(v) - float(branch["last_v"]))
                pred_gap = abs(float(v) - pred_v)
                local_scale = _branch_scale(branch)
                adaptive_tol = max(
                    float(tol),
                    1.4 * local_scale,
                    0.04 * max(abs(pred_v), abs(float(v)), 1.0),
                )
                adaptive_tol *= (1.0 + 0.25 * missed)

                freq_gap = abs(float(f) - float(branch["last_f"]))
                if freq_gap > (missed + 1.2) * max(base_df, 1.0e-8) * 1.8:
                    continue
                if pred_gap > adaptive_tol and last_gap > adaptive_tol:
                    continue

                score = pred_gap + 0.30 * last_gap + 0.12 * missed * adaptive_tol
                if best_score is None or score < best_score:
                    best_score = score
                    best_branch = branch

            if best_branch is None:
                branches.append({
                    "points": [(v, freq[src_idx])],
                    "last_v": v,
                    "last_f": freq[src_idx],
                    "missed": 0,
                    "closed": False,
                    "matched": True,
                })
            else:
                best_branch["points"].append((v, freq[src_idx]))
                best_branch["last_v"] = v
                best_branch["last_f"] = freq[src_idx]
                best_branch["missed"] = 0
                best_branch["matched"] = True

        for branch in branches:
            if branch["matched"]:
                continue
            branch["missed"] = int(branch.get("missed", 0)) + 1
            if branch["missed"] > max_skip_steps:
                branch["closed"] = True

    kept = []
    for branch in branches:
        pts = branch["points"]
        if len(pts) < min_points_per_mode:
            continue
        pts = np.array(pts, dtype=float)
        order_i = np.argsort(pts[:, 1])
        pts = pts[order_i]
        kept.append(pts)

    kept.sort(key=lambda pts: float(np.median(pts[:, 0])))
    if max_modes is not None:
        kept = kept[:max(1, int(max_modes))]

    if len(kept) == 0:
        return _single_mode_result(vel, freq)

    modes = {}
    for mode_id, pts in enumerate(kept):
        modes[mode_id] = np.vstack((pts[:, 0], pts[:, 1]))

    return modes

def detrend(arr): return signal.detrend(arr, axis=-1)


def taper(arr): return arr * np.hanning(arr.shape[-1])


def cut_cal_data(params, callback_func, callback_plotdata, callback_str, callback_str2, callback_str3, stop_flag):
    """
    load, cut and cal data
    Args:
        params: params
        callback_func:
        callback_plotdata:
        callback_str:
        callback_str2:
        callback_str3:
        stop_flag:

    Returns:

    """
    if params['files_jug'] == 'mul':  # 文件夹数目判断
        file = glob.glob(params['data_path'] + params['shape'])  # 遍历文件
        file.sort()
    else:
        file = glob.glob(params['data_path'])  # 遍历文件
        file.sort()
    tt_temp = False
    bt = time.time()
    val = params['samplerate_in_use']
    # params['samplerate'] = 200
    if (
            isinstance(val, (int, float))  # 是数字
            and not isinstance(val, bool)  # 不允许 bool
            and val != params['samplerate']  # 数值不同才触发
    ):
        de_factor = int(params['samplerate_in_use']) / int(params['samplerate'])
        det = 1 / de_factor
        # det = 1
        if de_factor == 1:
            pass
        if not det.is_integer():
            raise 'samplerate must be an integer multiple of samplerate_in_use !'
        b, a = signal.butter(4, [0.001, de_factor / 2], 'band')
        tt_temp = True
    elif not params['samplerate_in_use'] or isinstance(params['samplerate_in_use'], str):
        params['samplerate_in_use'] = params['samplerate']
    # file.sort()
    params['samplerate_in_use'] = float(params['samplerate_in_use'])
    cc_len = params['cc_len']
    single_time_len = params['time_len']
    corr_all_stack = []
    # free_size = get_memory() / ratio
    free_size = get_memory()  # 显存空闲（Mb）
    if params['meta_flag']:
        x, y = load_and_project(params['meta_file'])
    if params['datatype'] == 'h5' or params['datatype'] == 'H5':
        if params['files_jug'] == 'mul':
            file_list = get_file_list(file)
        else:
            file_list = sorted(glob.glob(f"{file[0]}*h5"))

        file_list.sort()
        if not file_list:
            raise FileNotFoundError(
                f"No HDF5 files matched data_path={params.get('data_path')!r}, "
                f"shape={params.get('shape')!r}"
            )
        f = h5py.File(file_list[0], 'r')
        data_struct = 'Acquisition/Raw[0]/RawData'
        data_struct = finddatastruct(f)
        data_size, params_temp = read_data(file_list[0], data_struct, para=True, data_temp=True)
        read_channel_range = params.get('read_channel_range')
        if read_channel_range is not None:
            read_b_ch, read_e_ch = map(int, read_channel_range)
            available_channels = int(f[data_struct].shape[1])
            if not (0 <= read_b_ch < read_e_ch <= available_channels):
                raise ValueError(
                    f"read_channel_range must be within [0, {available_channels}], "
                    f"got {read_channel_range!r}"
                )
            num_sta = read_e_ch - read_b_ch
        else:
            read_b_ch, read_e_ch = None, None
            num_sta = len(data_size)
        f.close()
        callback_str3(params_temp['starttime'])
    elif params['datatype'] == 'tdms' or params['datatype'] == 'TDMS':
        file_t = [file[1]]
        if params['files_jug'] == 'mul':
            file_list = get_file_list_tdms(file_t)
        else:
            file_list = sorted(glob.glob(f"{file_t}/*tdms"))
        file_list.sort()
        # num_sta = 4000
    # memory_in_bytes = len(data_size) * single_time_len * params['samplerate_in_use']
    # memeo_need_single = memory_in_bytes / (1024 ** 2)
    # xishu = cc_len / params['time_len']
    # memeo_need_single *= xishu
    # numfiles = free_size / (2 * memeo_need_single)
    # if numfiles >= len(file_list):
    # numfiles = len(file_list)
    samples_per_file = round(params['time_len']) * int(params['samplerate_in_use'])
    # max_files = int(free_size / (num_sta * samples_per_file * 4 / (1024 ** 2))) // 2  # 安全系数
    max_files = 60
    num_files = min(len(file_list), max_files)
    if params['datatype'] == 'h5' or 'H5':
        data_all = np.zeros((num_sta, num_files, samples_per_file), dtype=np.float32)
    max_threads = os.cpu_count()  #

    batch_size = max_files
    total_batches = max(int(np.ceil(len(file_list) / batch_size)), 1)
    callback_str('Loading data...')
    print(('Loading data...'))
    samplerate_org = params['samplerate']
    len_sta = 0
    temp = 0
    # data_sta_t = np.zeros((3, num_files, samples_per_file), dtype=np.float32)
    for il in range(0, len(file_list), batch_size):
        batch_number = il // batch_size
        batch_span = 90.0 / total_batches
        batch_start = 5.0 + batch_number * batch_span
        loaded_progress = batch_start + 0.25 * batch_span
        preprocess_progress = batch_start + 0.35 * batch_span
        correlation_end = batch_start + 0.90 * batch_span
        batch_end = batch_start + batch_span
        print(il)
        callback_func(int(round(batch_start)))
        if params['datatype'] == 'h5' or 'H5':

            def process_single_file(file_t, num):
                et_temp = None
                st_temp = None
                if not stop_flag():
                    return
                # for num in range(len(file_list[il:il + batch_size])):
                """处理单个HDF5文件"""
                with h5py.File(file_t, 'r') as f:  # 上下文管理器自动管理文件
                    # data = read_data(file_list[num], data_struct)  # 假设read_data已优化
                    if read_b_ch is None:
                        data = f[data_struct][:]
                    else:
                        data = f[data_struct][:, read_b_ch:read_e_ch]
                    data = np.array(data).transpose()
                    # 向量化滤波和下采样
                    if len(data[-1]) > params['time_len'] * samplerate_org + 1000:
                        data = signal.decimate(data, 4, axis=1, zero_phase=True)
                        len_temp = params['time_len'] * samplerate_org * 1 - len(data[-1])
                    else:
                        len_temp = params['time_len'] * samplerate_org * 1 - len(data[-1])
                    if len_temp > 0:
                        data = np.pad(data, ((0, 0), (0, int(len_temp) + 1)), 'constant', constant_values=0)
                    if tt_temp:
                        # data = signal.filtfilt(b, a, data, axis=1)
                        data_all[len_sta:, num, :] = signal.decimate(data, int(1 / de_factor), ftype='iir', axis=1, zero_phase=True)
                        # data_all_t = data_all
                        # data_all[:, num, :] = signal.medfilt(data_all[:, num, :])
                        # data_all[:, num, :] = fk_f(data_all[:, num, :], dt=0.02, dx=8.18, w=20, show=False, freq_window=5)
                    else:
                        data_all[:, num, :] = data
                        # data_all[:, num, :] = signal.medfilt(data)
                    if num == batch_size - 1:
                        params_temp = read_data(file_t, data_struct, para=True, data_temp=False)
                        # callback_str2(params_temp['endtime'])
                        # et_temp = params_temp['endtime']
                    if num == 0:
                        params_temp = read_data(file_t, data_struct, para=True, data_temp=False)
                        st_temp = params_temp['starttime']
                # return et_temp, st_temp

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(process_single_file, file, ibch)
                           for ibch, file in enumerate(file_list[il:il + batch_size])]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                    pass
        elif params['datatype'] == 'tdms' or params['datatype'] == 'TDMS':
            if TdmsFile is None:
                raise RuntimeError(
                    "TDMS support requires the optional 'nptdms' package. "
                    "Install it before processing TDMS files."
                )
            data_all = np.zeros((num_sta, num_files, samples_per_file), dtype=np.float32)

            # def process_single_group(tdms_path, num):
            for num, tdms_path in enumerate(file_list[il:il + batch_size]):
                """
                线程函数：读取 TDMS 中一个 group
                """
                tdms = TdmsFile.read(tdms_path)

                # ---------- 1. 找有效 group ----------
                valid_groups = [g for g in tdms.groups() if len(g.channels()) > 0]
                if not valid_groups:
                    raise RuntimeError("No TDMS group containing data was found")

                # ---------- 2. 用第一个 group 定义 channel 集合 ----------
                ref_group = valid_groups[0]
                ch_names = [ch.name for ch in ref_group.channels()]
                nch = len(ch_names)
                nt_min = int(round(params['time_len'])*params['samplerate'])

                # ---------- 3. 预分配 ----------
                nt_total = nt_min * len(valid_groups)
                data = np.full((nch, nt_total), 0, dtype=np.float32)

                # ---------- 4. 按 channel 名对齐拼接 ----------
                t_index = 0
                for g in valid_groups:
                    print(g.name, len(g.channels()))

                    t0 = t_index * nt_min
                    t1 = t0 + nt_min

                    ch_dict = {ch.name: ch for ch in g.channels()}

                    for ich, name in enumerate(ch_names):
                        if name in ch_dict:
                            if len(data[ich, t0:t1]) == len(ch_dict[name][:]):
                                data[ich, t0:t1] = ch_dict[name][:]
                            elif len(data[ich, t0:t1]) > len(ch_dict[name][:]):
                                data[ich, t0:t1] = np.concatenate((ch_dict[name][:], np.zeros((len(data[ich, t0:t1])-len(ch_dict[name][:])))))
                            else:
                                data[ich, t0:t1] = ch_dict[name][:]
                        else:
                            # 缺失 channel：保持 fill_value
                            pass

                    t_index += 1
                if tt_temp:
                    data_all[len_sta:, num, :] = signal.decimate(data, 2, n=None, ftype='iir', axis=-1, zero_phase=True)
                else:
                    data_all[:, num, :] = data
                print(num)
            #     return 0,0
            # with ThreadPoolExecutor(max_workers=max_threads) as executor:
            #     # results = list(tqdm(executor.map(process_single_file, file_list[il:il+batch_size], range(batch_size))))  # 并行处理
            #     with tqdm(total=len(file_list[il:il + batch_size]), desc="Processing files") as pbar:
            #         futures = []
            #         for ibch, file in enumerate(file_list[il:il + batch_size]):
            #             future = executor.submit(process_single_group, file, ibch)

        if not stop_flag():
            return
        time_len = 0
        idx = 0
        e = time.time()
        print('load_lost:{%f}' % (e - bt))
        params['samplerate'] = params['samplerate_in_use']
        params['delta'] = 1 / params['samplerate']

        # finallist = [x for x in results if x is not None]
        # filtered_list = [
        #     tuple(item for item in tpl if item is not None)
        #     for tpl in results
        #     if any(item is not None for item in tpl)  # 可选：移除全为 None 的元组
        # ]
        #
        # st_temp, et_temp = filtered_list
        # st_temp = st_temp[0]
        # et_temp = et_temp[0]
        #
        # if params['STA']:
        #     date = st_temp.date.strftime("%Y%m%d")
        #     data_STA = obspy.read(params['sta_file'] + '/' + date + '000000/SD*BHN')
        #     st_temp = st_temp - 8 * 60 * 60
        #     et_temp = et_temp - 8 * 60 * 60
        #     data_STA = data_STA.slice(st_temp, et_temp)
        #     data_STA = data_STA.decimate(factor=2)
        #     data_sta = np.array(data_STA)
        #     if data_sta.shape[-1] >= 180000:
        #         data_sta_t = data_sta[:, :180000]
        #     elif data_sta.shape[-1] < 180000:
        #         data_sta_t[:, :data_sta.shape[-1]] = data_sta
        #     data_sta = data_sta_t.reshape(data_sta_t.shape[0], num_files, samples_per_file)
        #     if len(data_all) == num_sta:
        #         data_all = np.vstack((data_sta, data_all))
        #     else:
        #         data_all[:len(data_sta)] = data_sta
        #     len_sta = len(data_STA)
        #     if len_sta == 0:
        #         break
        callback_func(int(round(loaded_progress)))
        if round(single_time_len) != cc_len:
            step = params['time_step']
            data_all_t = data_all.reshape(data_all.shape[0], -1)

            sps = params['samplerate']
            npts = int(cc_len * sps)
            step_samples = int(step * sps)  # 步长对应的样本数

            num_segments = (data_all_t.shape[1] - npts) // step_samples + 1
            # 预分配内存
            if data_all.shape[1] < params['samplerate'] * time_len:
                print("ERROR! Data array is short than theoretical values!")
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                idx += 1
                continue
            step_samples = int(step * sps)
            segments = np.lib.stride_tricks.sliding_window_view(data_all_t, npts, axis=1)[:, ::step_samples, :]
            dataS = segments[:, :num_segments, :]
            epsilon = 1e-6
            all_stdS = allstd_global(dataS)
            all_stdS += epsilon
            dataS, trace_stdS = numba_process1(dataS, all_stdS)
            e1 = time.time()
            print('pre_lost:{%f}' % (e1 - e))
        else:
            print('Calculating Astd')
            if data_all.shape[1] * data_all.shape[-1] < params['samplerate_in_use'] * time_len:
                print("ERROR! Data array is short than theoretical values!")
                data_all = []
                temp_size = 0
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                idx += 1
                continue
            all_stdS = allstd_global(data_all)
            print('Data preprocessing ')
            dataS, trace_stdS = numba_process1(data_all, all_stdS)
            e1 = time.time()
            print('pre_lost:{%f}' % (e1 - e))
        callback_str('Calculating cross-correlation...')
        callback_func(int(round(preprocess_progress)))
        sou_ind = np.zeros((trace_stdS.shape[0], trace_stdS.shape[1]))
        for i in range(len(trace_stdS)):
            sou_ind[
                i, np.where((trace_stdS[i, :] < 10) & (trace_stdS[i, :] > 0) & (np.isnan(trace_stdS[i, :]) == 0))[
                    0]] = 1
        if not stop_flag():
            return
        corr_all_t = corrs_cor(
            dataS,
            params,
            sou_ind,
            callback_func,
            progress_start=preprocess_progress,
            progress_end=correlation_end,
        )
        callback_func(int(round(correlation_end)))
        e2 = time.time()
        print('cc_lost:{%f}' % (e2 - e1))
        # print(corr_all_t)
        if corr_all_t[0].shape[-1] != params['samplerate_in_use'] * params['cc_len']:
            # corr_all_t = cp.array(corr_all_t)
            corr_all_t = signal.decimate(cp.asnumpy(cp.array(corr_all_t)),
                                         int(corr_all_t[0].shape[-1] * 2 / (
                                                 params['samplerate_in_use'] * params['cc_len'])))
        # task2 = asyncio.create_task(conc(corr_all, corr_all_t))
        # x = cp.asarray(x)
        # y = cp.asarray(y)
        # for stai in range(len(corr_all_t)):
        #     det_azs, all_beam_power, slowness, azimuths, slowness_indx = beamforming_analysis_gpu(
        #         cp.asarray(corr_all_t[stai][1:]), cp.asarray(x[stai * params['step']:stai * params[
        #             'step'] + params['c_range']]), cp.asarray(y[
        #                                                       stai * params['step']:stai * params['step'] + params[
        #                                                           'c_range']]),
        #         params['samplerate_in_use'], win_len=int(cc_len * params['samplerate_in_use'] / 2), n_windows=2)
        #     if not (0 < det_azs[-1] < 30) and not (330 < det_azs[-1] < 360) and not (150 < det_azs[-1] < 210):
        #         corr_all_t[stai][1:] = 0
        #         print(stai)
        #         print(det_azs[-1])
        #
        #     if il % 10 == 0 and stai == 200:
        #         plot_beamforming_results(all_beam_power, slowness, azimuths, slowness_indx, det_azs, il)

        # if (il) % 1 == 0:
        #     data_temp2save = np.array(cp.array(corr_all_t).squeeze().get())
        #     print(il)
        if il == 0 and idx == 0:
            corr_all = cp.expand_dims(cp.array(corr_all_t), axis=-2)
            corr_all_stack = corr_all
        else:
            if os.path.exists(params['outputdir'] + f'/cc.h5') and temp == 0:
                corr_all_stack, _ = np.array(read_single_h5(params['outputdir'] + f'/cc.h5'))
                corr_all_stack = cp.expand_dims(cp.array(corr_all_stack), axis=-2)
                temp = 1
            corr_all = cp.concatenate((corr_all_stack, cp.expand_dims(cp.array(corr_all_t), axis=-2)),
                                      axis=-2)
            for sor in range(len(corr_all)):
                # corr_all_stack[sor] = cp.expand_dims(pws_time_domain_new(corr_all[sor], power=2), axis=-2)
                corr_all_stack[sor] = cp.expand_dims(linear_stack_gpu(corr_all[sor]), axis=-2)
        # callback_plotdata(np.array(cp.array(corr_all_t).get()))
        callback_plotdata(np.array(cp.squeeze(corr_all_stack, axis=-2).get()))

        # del x
        # del y
        del corr_all
        del corr_all_t

        callback_func(int(round(batch_end)))
        data_temp2save = np.array(cp.squeeze(corr_all_stack, axis=-2).get())
        ncf_dir = os.path.join(params['outputdir'], 'NCF')
        os.makedirs(ncf_dir, exist_ok=True)
        write_single_h5(os.path.join(ncf_dir, 'cc2.h5'), data_temp2save, params=params)
        res_write(data_temp2save, params)
        # if (il) % 6 == 0:
        #     data_temp2save = np.array(corr_all_stack.squeeze().get())
        #     write_single_h5(params['outputdir'] + '/NCF' + f'cc{il}.h5', data_temp2save)
        #     if not os.path.exists(params['outputdir'] + '/NCF'):
        #         os.mkdir(params['outputdir'] + '/NCF')
        #     else:
        #         shutil.rmtree(params['outputdir'] + '/NCF')
        #         os.mkdir(params['outputdir'] + '/NCF')
        #     if params['outtype'] == 'h5' or params['outtype'] == 'H5':
        #         write_single_h5(params['outputdir'] + '/NCF' + f'cc{il}.h5', data_temp2save)
        #     elif params['outtype'] == 'SAC' or params['outtype'] == 'sac':
        #         res_write(data_temp2save, params)
        #     else:
        #         write_single_h5(params['outputdir'] + '/' + f'cc_1.h5', data_temp2save, params)
        #         res_write(data_temp2save, params)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        idx += 1
    callback_func(100)
    return cp.squeeze(corr_all_stack, axis=-2)


def calculate_chunk_size(data_shape, dtype_size=4, safety_margin=0.2):
    """
    根据显存容量动态计算最大分块大小

    参数:
        data_shape: 数据块的形状 (典型为 (chunk_size, channels, samples))
        dtype_size: 数据类型大小（字节），float32为4
        safety_margin: 安全余量比例

    返回:
        max_chunk_size: 最大可分块数量
    """
    # 获取GPU显存信息
    mem_info = cp.cuda.runtime.memGetInfo()
    free_mem = mem_info[0]  # 空闲显存（字节）

    # 计算单个数据元素的内存占用
    element_size = dtype_size
    elements_per_chunk = np.prod(data_shape[1:])  # channels * samples

    # 计算最大分块数量（保守估计）
    max_elements = (free_mem * (1 - safety_margin)) // element_size
    max_chunk_size = max_elements // elements_per_chunk

    return int(max_chunk_size)


def prepare_memmap(data_all, filename='data.dat'):
    mmap = np.memmap(filename, dtype='float32', mode='w+', shape=data_all.shape)
    mmap[:] = data_all[:]
    return mmap


def ncf_slced(filename):
    files = glob.glob(filename)

    def slc(i):
        data_raw = read_single_h5(files[i])


def corrs_cor(
        data_all,
        cal_params,
        sou_ind,
        callback_func,
        progress_start=35.0,
        progress_end=95.0,
):
    c_range = cal_params['c_range']
    step = cal_params['step']
    corr_all = []
    # corr_all_t = []
    # corr_temp_stacked = []
    # if data_all.ndim <= 2:
    #     data_all = cp.expand_dims(data_all, axis=1)
    Nfft = data_all.shape[-1]
    global_results = []
    sou_ind_t = np.empty_like(sou_ind)
    Nfft2 = Nfft // 2
    # correlation_matrix_1 = cp.ones((c_range + 1, data_all.shape[-2]))
    # correlation_matrix_2 = cp.ones((c_range + 1, data_all.shape[-2]))
    # if not os.path.exists('./data.dat'):
    #     data_mmap = prepare_memmap(data_all)
    # else:
    #     data_mmap = np.memmap('data.dat', dtype='float32', mode='r', shape=data_all.shape)
    source_indices = list(range(
        int(cal_params['b_ch']),
        int(cal_params['e_ch']) - c_range - 1,
        step,
    ))
    total_sources = max(len(source_indices), 1)
    with cp.cuda.Stream() as main_stream:
        main_stream.synchronize()  # 确保计算完成

        # 动态计算初始分块大小（假设 calculate_chunk_size 已定义）
    chunk_size = calculate_chunk_size((1000, data_all.shape[1], data_all.shape[2]))
    current_chunk = 0
    while current_chunk * chunk_size < len(data_all):
        start = current_chunk * chunk_size
        end = start + chunk_size

        # 用默认流进行数据加载（同步方式，但在流上下文中）
        with cp.cuda.Stream() as stream:
            data_block = cp.asarray(data_all[start:end])
            # 在当前流下执行数据处理
            results = []
            for source_number, i in enumerate(source_indices):
                print(i)
                # if cal_params['STA']:

                fraction = (source_number + 1) / total_sources
                progress = progress_start + (progress_end - progress_start) * fraction
                callback_func(int(round(progress)))
                for it in range(i, i + c_range + 1):
                    # sou_ind[it] = np.intersect1d(sou_ind[i], sou_ind[it])
                    sou_ind_t[it] = sou_ind[i] * sou_ind[it]
                # 获取当前窗口数据
                window = data_block[i:i + c_range + 1]
                # window = data_block
                # 数据标准化：先计算绝对值，再通过 moving_ave_cov 计算平滑因子
                # abs_data = cp.abs(window)
                # norm_factor = moving_ave_cov(abs_data, cal_params['smooth_N'])
                # window /= norm_factor

                # FFT 计算
                window = suppress_resonances_in_spectra(window, smooth_win=11, thresh_mult=3, expand_bins=2)
                FFTRawSign = cp.fft.fft(window, Nfft)
                # 释放 data_norm 内存
                del window
                cp.get_default_memory_pool().free_all_blocks()

                # 噪声处理（假设 noise_processing 在 GPU 上运行）
                receiver_white = noise_processing(cal_params, FFTRawSign)
                del FFTRawSign
                # abs_data
                cp.get_default_memory_pool().free_all_blocks()
                # 交叉相关计算
                if cal_params['substack']:
                    # receiver_white = cp.fft.ifft(receiver_white)[:,:,::-1]
                    # corr_temp = cross_convolve_with_first_station(receiver_white)
                    receiver_white *= cp.array(sou_ind_t[i:i + c_range + 1, :, cp.newaxis])
                    # corr_temp = xcorr(receiver_white, Nfft2, i)
                    corr_temp = xcorr_by_offset_concat_seg_chunk(receiver_white, Nfft2)
                    corr_temp = pws_time_domain_list(corr_temp)
                    # corr_temp = xcoherence(receiver_white, Nfft2, i)
                    # 利用广播：对当前窗口对应的 sou_ind_t 分块进行乘法
                    # 假设 sou_ind_t 的形状能通过索引 [i:i+c_range+1, :, cp.newaxis] 得到正确的广播结果
                    # corr_temp *= sou_ind_t[i:i + c_range + 1, :, cp.newaxis]
                    corr_temp = cp.asarray(corr_temp)
                    # corr_temp *= cp.array(sou_ind_t[i:i + c_range + 1, :, cp.newaxis])
                    ##cc_save
                    # if not hasattr(corrs_cor, "has_run"):
                    #     corr_temp_s = np.array(cp.array(corr_temp).get())
                    #                     cal_params
                    #                     )
                    #     corrs_cor.has_run = True  # 设置标志位

                    # correlation_matrix_1[1:] = vectorized_3d_corr(
                    #     corr_temp[1:, :, int(corr_temp.shape[-1] / 2) + 10:int(corr_temp.shape[-1] / 2) + 6 + 150],
                    #     corr_temp[:-1, :, int(corr_temp.shape[-1] / 2) + 10:int(corr_temp.shape[-1] / 2) + 6 + 150])
                    # correlation_matrix_2[2:] = vectorized_3d_corr(
                    #     corr_temp[2:, :, int(corr_temp.shape[-1] / 2) + 10:int(corr_temp.shape[-1] / 2) + 6 + 150],
                    #     corr_temp[:-2, :, int(corr_temp.shape[-1] / 2) + 10:int(corr_temp.shape[-1] / 2) + 6 + 150])
                    # correlation_matrix_2[correlation_matrix_2 > 0.6] = 1
                    # correlation_matrix_2[correlation_matrix_2 <= 0.6] = 0
                    # correlation_matrix_1[correlation_matrix_1 > 0.75] = 1
                    # correlation_matrix_1[correlation_matrix_1 <= 0.75] = 0
                    # correlation_matrix_2 = cp.nan_to_num(correlation_matrix_2, nan=0.0)
                    # correlation_matrix_1 = cp.nan_to_num(correlation_matrix_1, nan=0.0)
                    # # correlation_matrix = cp.multiply(correlation_matrix_1, correlation_matrix_2)
                    # correlation_matrix = correlation_matrix_1+correlation_matrix_2
                    # corr_temp = correlation_matrix[..., cp.newaxis]*corr_temp
                    # if i == 800:
                    #     data_temp2save = np.array(cp.array(corr_temp[:, :, ::10]).get())
                    # result = pws_time_domain_new(corr_temp)
                    # result = linear_stack_gpu(corr_temp)
                    results.append(corr_temp)

                else:
                    corr_temp = xcorr(receiver_white, Nfft2, i)
                    corr_temp *= cp.array(sou_ind_t[i:i + c_range + 1, :, cp.newaxis])
                    corr_temp = cp.asarray(corr_temp)
                    result = pws_time_domain_new(corr_temp)
                    # result = linear_stack_gpu(corr_temp)
                    results.append(result)

                del receiver_white
                cp.get_default_memory_pool().free_all_blocks()

            # 同步等待当前流完成所有操作
            stream.synchronize()
            global_results.extend(results)
            # # 异步写回 CPU 内存
            # cpu_results = [cp.asnumpy(r) for r in results]

        current_chunk += 1
        del data_block, results
    return global_results


def fft_convolve_cupy(x, y):
    n = len(x) + len(y) - 1
    nfft = 2 ** int(cp.ceil(cp.log2(n)))

    fft_x = cp.fft.fft(x, nfft)
    fft_y = cp.fft.fft(y, nfft)

    conv = cp.fft.ifft(fft_x * fft_y)
    return cp.asnumpy(cp.real(conv[:n]))


def cross_convolve_with_first_station(data):
    """
    data: numpy.ndarray, shape (n_stations, n_segments, n_times)

    返回:
    result: numpy.ndarray, shape (n_stations - 1, n_segments, 2*n_times -1)
    对第一个台站与其他台站在时间维度做卷积。
    """
    n_stations, n_segments, n_times = data.shape
    result = np.zeros((n_stations, n_segments, 2 * n_times - 1), dtype=cp.float32)

    first_station = data[0, :, :]  # shape (n_segments, n_times)

    for i in range(n_stations):
        for j in range(n_segments):
            x = first_station[j, :]
            y = data[i, j, :]
            # 这里是卷积，如果你想用自相关，请把 y 改为 y[::-1]
            result[i, j, :] = fft_convolve_cupy(x, y)
    return result


def vectorized_corr(A, B):
    # 中心化数据 (减去行均值)
    A_centered = A - np.mean(A, axis=1, keepdims=True)
    B_centered = B - np.mean(B, axis=1, keepdims=True)

    # 计算分子 (协方差)
    cov = np.sum(A_centered * B_centered, axis=1)

    # 计算分母 (标准差乘积)
    std_A = np.sqrt(np.sum(A_centered ** 2, axis=1))
    std_B = np.sqrt(np.sum(B_centered ** 2, axis=1))

    # 计算相关系数并处理除零错误
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / (std_A * std_B)
        corr[std_A == 0] = np.nan
        corr[std_B == 0] = np.nan

    return corr


def vectorized_3d_corr(A, B):
    """
    高效计算两个三维数组的对应位置向量的相关系数
    参数:
    A : numpy.ndarray, 形状 (n-1, d, 45)
    B : numpy.ndarray, 形状 (n-1, d, 45)
    返回:
    corr_matrix : numpy.ndarray, 形状 (n-1, d)
        相关系数矩阵，corr_matrix[i, j] 表示 A[i, j] 和 B[i, j] 的相关系数
    """
    # 1. 中心化数据 (减去每个向量的均值)
    A_centered = A - cp.mean(A, axis=2, keepdims=True)
    B_centered = B - cp.mean(B, axis=2, keepdims=True)
    # 2. 计算协方差 (点积)
    cov = cp.sum(A_centered * B_centered, axis=2)
    # 3. 计算每个向量的L2范数 (标准差)
    std_A = cp.sqrt(np.sum(A_centered ** 2, axis=2))
    std_B = cp.sqrt(np.sum(B_centered ** 2, axis=2))
    # 4. 计算相关系数并处理除零错误
    temp = (std_A * std_B)
    corr = cov / temp
    # 处理标准差为零的情况
    zero_std = (std_A == 0) | (std_B == 0)
    corr = cp.where(zero_std, cp.nan, corr)
    # denominator = std_A * std_B
    #
    # # 创建初始结果数组
    # corr = cp.empty_like(cov)
    #
    # # 安全除法
    # valid_mask = denominator != 0
    # corr[valid_mask] = cov[valid_mask] / denominator[valid_mask]
    #
    # # 处理分母为零的情况
    # zero_mask = ~valid_mask
    # corr[zero_mask] = float('nan')
    return corr


def moving_ave_cov(A, N):
    '''
    this Numba compiled function does running smooth average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the half window length to smooth

    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    '''
    A = cp.concatenate((A[:, :, :N], A, A[:, :, -N:]), axis=-1)
    # B = cp.zeros(A.shape, A.dtype)
    kernel = (cp.ones(2 * N, dtype=A.dtype) / (2 * N + 1))
    cp.get_default_memory_pool().free_all_blocks()
    B = nd.convolve(A, kernel[None, None, :], mode='constant', cval=0.0)
    A = None
    cp.get_default_memory_pool().free_all_blocks()
    B[B == 0] = 1
    # if B[:, :, pos] == 0:
    #    B[:, :, pos] = 1

    return B[:, :, N:-N]


def noise_processing(fft_para, FFTRawSign):
    # load parameters first
    delta = fft_para['delta']
    freqmin = cp.float32(fft_para['freqmin'])
    freqmax = fft_para['freqmax']
    smooth_N = fft_para['smooth_N']
    freq_norm = fft_para['freq_norm']
    freqmin = cp.asarray(freqmin)
    freqmax = cp.asarray(freqmax)
    delta = cp.asarray(delta)
    Nfft = int(next_fast_len(int(FFTRawSign.shape[2])))
    Nfft = int(Nfft)
    freqVec = cp.fft.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = cp.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    Napod = 100
    low = J[0] - Napod
    if low <= 0:
        low = 1
    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = int(Nfft // 2)
    FFTRawSign[:, :, 0:low] *= 0
    FFTRawSign[:, :, low:left] = cp.cos(
        cp.linspace(np.pi / 2., cp.pi, int(left - low))) ** 2 * cp.exp(
        1j * cp.angle(FFTRawSign[:, :, low:left]))
    # Pass band:
    if freq_norm == 'phase_only':
        FFTRawSign[:, :, left:right] = cp.exp(1j * cp.angle(FFTRawSign[:, :, left:right]))
    # elif freq_norm == 'rma':
    else:
        tave = moving_ave_cov(cp.abs(FFTRawSign[:, :, left:right]), smooth_N)
        FFTRawSign[:, :, left:right] = FFTRawSign[:, :, left:right] / tave
    # Right tapering:
    FFTRawSign[:, :, right:high] = cp.cos(
        cp.linspace(0., cp.pi / 2., int(high - right))) ** 2 * cp.exp(
        1j * cp.angle(FFTRawSign[:, :, right:high]))
    FFTRawSign[:, :, high:Nfft // 2] *= 0

    # Hermitian symmetry (because the input is real)
    FFTRawSign[:, :, -(Nfft // 2) + 1:] = cp.flip(cp.conj(FFTRawSign[:, :, 1:(Nfft // 2)]), axis=-1)
    return FFTRawSign


def single_pws(arr, sampling_rate=250, power=2, pws_timegate=5.):
    """
     Performs phase-weighted stack on array of time series. Modified on the noise function by Tim Climents.
     Follows methods of Schimmel and Paulssen, 1997.
     If s(t) is time series data (seismogram, or cross-correlation),
     S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
     S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
     A(t) is envelope of s(t) and phi(t) is phase of s(t)
     Phase-weighted stack, g(t), is then:
     g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
     where N is number of traces used, v is sharpness of phase-weighted stack

     PARAMETERS:
     ---------------------
     arr: N length array of time series data (numpy.ndarray)
     sampling_rate: sampling rate of time series arr (int)
     power: exponent for phase stack (int)
     pws_timegate: number of seconds to smooth phase stack (float)

     RETURNS:
     ---------------------
     weighted: Phase weighted stack of time series data (numpy.ndarray)
     """

    if arr.ndim == 1:
        return arr
    N, M = arr.shape
    analytic = hilbert(arr, axis=1, N=next_fast_len(M))[:, :M]
    phase = cp.angle(analytic)
    phase_stack = cp.mean(np.exp(1j * phase), axis=0)
    phase_stack = cp.abs(phase_stack) ** (power)

    # smoothing
    # timegate_samples = int(pws_timegate * sampling_rate)
    # phase_stack = moving_ave(phase_stack,timegate_samples)
    weighted = cp.multiply(arr, phase_stack)
    return cp.mean(weighted, axis=0)


def hilbert(x, N=None, axis=-1):
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = cp.fft.fft(x, N, axis=axis)
    h = cp.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = cp.fft.ifft(Xf * h, axis=axis)
    return x
    pass


# def xcorr(r_data, Nfft2):
#     source_white = cp.conjugate(r_data[0])
#     corr = source_white * r_data
#     corr[:, :, :Nfft2] = corr[:, :, :Nfft2] - cp.mean(corr[:, :, :Nfft2], axis=-1)[:, :, cp.newaxis]
#     corr[:, :, -(Nfft2) + 1:] = cp.flip(cp.conj(corr[:, :, 1:(Nfft2)]), axis=-1)
#     corr[:, :, Nfft2] = complex(0, 0)
#     corr[:, :, 0] = complex(0, 0)
#     s_corr = cp.real(cp.fft.ifftshift(cp.fft.ifft(corr, Nfft2 * 2, axis=-1), axes=-1))
#
#     return s_corr


def xcorr(r_data, Nfft2, i):
    # Precompute the conjugate of the first element
    source_white = cp.conjugate(r_data[0])

    # Compute the correlation
    corr = source_white * r_data

    # Subtract the mean from the first Nfft2 elements
    mean_corr = cp.mean(corr[:, :, :Nfft2], axis=-1, keepdims=True)
    corr[:, :, :Nfft2] -= mean_corr

    # Handle the symmetry in the correlation
    corr[:, :, -Nfft2 + 1:] = cp.flip(cp.conj(corr[:, :, 1:Nfft2]), axis=-1)

    # Set the middle and first elements to zero
    corr[:, :, Nfft2] = 0
    corr[:, :, 0] = 0

    # Compute the inverse FFT and shift in one step
    corr = cp.ascontiguousarray(corr)
    # s_corr = cp.fft.ifftshift(cp.fft.ifft(corr, Nfft2 * 2, axis=-1), axes=-1)
    s_corr = cp.fft.ifftshift(cp.fft.ifft(corr, Nfft2 * 2, axis=-1), axes=-1)

    # Return the real part of the result
    return cp.real(s_corr)

def xcorr_by_offset_concat_seg_chunk(
    r_data, Nfft2, ch_block=8
):
    """
    ch_block: 每次处理多少对道（越小越省显存）
    """

    Nch, Nseg, _ = r_data.shape
    Nfreq = 2 * Nfft2

    out = []

    for k in range(Nch):
        if k == 0:
            src_all = r_data
            rec_all = r_data
        else:
            src_all = r_data[:-k]
            rec_all = r_data[k:]

        n_pair = src_all.shape[0]
        seg_list = []

        for i0 in range(0, n_pair, ch_block):
            i1 = min(i0 + ch_block, n_pair)

            src = src_all[i0:i1]     # (B, Nseg, Nfreq)
            rec = rec_all[i0:i1]

            corr = cp.conjugate(src) * rec

            mean_corr = cp.mean(
                corr[:, :, :Nfft2],
                axis=-1,
                keepdims=True
            )
            corr[:, :, :Nfft2] -= mean_corr

            corr[:, :, -Nfft2 + 1:] = cp.flip(
                cp.conj(corr[:, :, 1:Nfft2]),
                axis=-1
            )
            corr[:, :, Nfft2] = 0
            corr[:, :, 0] = 0

            corr = cp.ascontiguousarray(corr)
            s_corr = cp.fft.ifftshift(
                cp.fft.ifft(corr, Nfreq, axis=-1),
                axes=-1
            )

            # 立刻 reshape + 收集
            s_corr = cp.real(s_corr).reshape(-1, Nfreq)
            seg_list.append(s_corr)

            # 关键：主动释放中间变量引用
            del corr, s_corr

        # offset k 的所有块拼接
        out.append(cp.concatenate(seg_list, axis=0))

    return out



def xcoherence(r_data, Nfft2, i):
    # r_data: (virtual_shots, subarrays, freq)
    # r_data[0] 作为参考

    X1 = r_data[0]  # 参考频谱
    X2 = r_data  # 所有通道频谱

    # === Step 1: compute cross spectrum ===
    Sxy = cp.conjugate(X1) * X2

    # === Step 2: auto spectra ===
    Sxx = cp.abs(X1)
    Syy = cp.abs(X2)

    # === Step 3: coherence (0 ~ 1) ===
    eps = 1e-12
    coh = cp.abs(Sxy) / (Sxx * Syy + eps)  # 形状与原 corr 一样

    # === Step 4: 不做去均值、也不做强制对称化 ===
    # 互相干不需要这些

    # === Step 5: IFFT → 时域互相干函数 ===
    coh = cp.ascontiguousarray(coh)
    s_coh = cp.fft.ifftshift(
        cp.fft.ifft(coh, Nfft2 * 2, axis=-1),
        axes=-1
    )

    # === Step 6: 返回实部 ===
    return cp.real(s_coh)


def suppress_resonances_in_spectra(X, smooth_win=9, thresh_mult=5.0, expand_bins=2, eps=1e-12):
    """
    在频域掐除窄带共振并用插值替换（GPU/CuPy版本）。
    X: complex array with shape (..., nfreq)
    smooth_win: 用来估计基线的滑动平均窗口（必须是奇数）
    thresh_mult: 当 PSD > baseline * thresh_mult 时认为是共振峰
    expand_bins: 在每个检测到的峰周围额外扩展的 bin 数
    返回: X_corrected (same shape)
    """
    # nfreq
    nfreq = X.shape[-1]

    # 计算全局 PSD（对所有通道/记录平均），用于检测共振频率
    PSD_global = cp.mean(cp.abs(X) ** 2, axis=tuple(range(X.ndim - 1)))  # shape (nfreq,)

    # 平滑 PSD 估计基线（用简单的卷积平均）
    w = smooth_win if smooth_win % 2 == 1 else smooth_win + 1
    kernel = cp.ones(w, dtype=PSD_global.dtype) / w
    baseline = cp.convolve(PSD_global, kernel, mode='same')

    # 检测尖峰
    peaks = PSD_global > (baseline * thresh_mult)

    # 扩展峰（dilate）
    if expand_bins > 0:
        kern_expand = cp.ones(expand_bins * 2 + 1, dtype=cp.int8)
        peaks_int = peaks.astype(cp.int8)
        peaks_dil = cp.convolve(peaks_int, kern_expand, mode='same') > 0
    else:
        peaks_dil = peaks

    # 构造 mask：True 保留点，False 被掐点
    mask_keep = ~peaks_dil

    # 如果没有检测到峰，直接返回原谱
    if cp.all(mask_keep):
        return X

    # 需要对每个被掐点用邻域未掐点插值替换 PSD 估计，然后对谱按幅度缩放
    freq_idx = cp.arange(nfreq)

    # good indices & values (on GPU)
    good_idx = cp.where(mask_keep)[0]
    good_vals = PSD_global[good_idx]

    # 若 good_idx 太少（例如全被掐），直接进行全谱平滑替代
    if good_idx.size < 2:
        # fallback: 用平滑 baseline 替代
        replacement_PSD = baseline
    else:
        # 使用 cupy.interp 在 GPU 上做线性插值
        # cupy.interp 存在并兼容 numpy.interp 接口
        replacement_PSD = cp.interp(freq_idx, good_idx, good_vals)

    # 现在对所有通道的频谱做幅度修正：
    # 想法：对每个频点计算缩放因子 s(f) = sqrt( replacement_PSD(f) / (PSD_global(f) + eps) )
    # 然后对 X 的每个频点乘以 s(f)（保持相位）
    scale = cp.sqrt(replacement_PSD / (PSD_global + eps))  # shape (nfreq,)

    # 为广播调整 shape
    # 例如 X.shape = (shots, subarrays, nfreq) -> scale shape (1,1,nfreq) 或 (1,...,nfreq)
    expand_shape = [1] * X.ndim
    expand_shape[-1] = nfreq
    scale = scale.reshape(expand_shape)

    X_corrected = X * scale

    return X_corrected


def xcorr_all(r_data, Nfft2, vs):
    # source_white = cp.conjugate(r_data)
    # corr = cp.einsum('ij,ik->ijk', source_white, r_data.T).T  #要阵列乘
    source_white = cp.conjugate(r_data[vs])
    corr = source_white * r_data
    corr[:, :, :Nfft2] = corr[:, :, :Nfft2] - cp.mean(corr[:, :, :Nfft2], axis=-1)[:, :, cp.newaxis]
    corr[:, :, -(Nfft2) + 1:] = cp.flip(cp.conj(corr[:, :, 1:(Nfft2)]), axis=-1)
    corr[:, :, Nfft2] = complex(0, 0)
    corr[:, :, 0] = complex(0, 0)
    s_corr = cp.real(cp.fft.ifftshift(cp.fft.ifft(corr, Nfft2 * 2, axis=-1), axes=-1))

    return s_corr


def xcorr_c2(r_data, Nfft2, sou_ind, samplerate=250):
    s_corr_s = []
    for i in range(len(r_data)):
        source_white = cp.conjugate(r_data[i])
        corr = source_white * r_data
        corr[:, :, :Nfft2] = corr[:, :, :Nfft2] - cp.mean(corr[:, :, :Nfft2], axis=-1)[:, :, cp.newaxis]
        corr[:, :, -(Nfft2) + 1:] = cp.flip(cp.conj(corr[:, :, 1:(Nfft2)]), axis=-1)
        corr[:, :, Nfft2] = complex(0, 0)
        corr[:, :, 0] = complex(0, 0)
        s_corr = cp.real(cp.fft.ifftshift(cp.fft.ifft(corr, Nfft2 * 2, axis=-1), axes=-1))
        shortest_index = min(range(len(sou_ind)), key=lambda i: len(sou_ind[i]))
        longest_index = max(range(len(sou_ind)), key=lambda i: len(sou_ind[i])) * 2 / 3
        minlen = len(sou_ind[shortest_index])
        maxlen = round(len(sou_ind[longest_index]) * 2 / 3)
        if minlen < maxlen:
            for it in range(len(s_corr)):
                s_corr_s.append(single_pws(s_corr[it, sou_ind[100 + it], :], sampling_rate=250))
        else:
            s_corr = pws_time_domain_new(s_corr[:, sou_ind[shortest_index], :]).get()
            s_corr_t = cp.zeros_like(s_corr)
            s_corr_t[:len(s_corr) - i, :] = s_corr[i:, :]
            s_corr_s.append(s_corr_t)
    s_corr_s = cp.array(s_corr_s)
    s_corr_s = s_corr_s.reshape(corr.shape[0], corr.shape[0], corr.shape[-1])
    s_corr_s = s_corr_s.transpose(1, 0, 2)
    s_corr_f = pws_time_domain_new(s_corr_s[:, :, :]).get()
    return s_corr_f


def linear_stack_gpu(arr, weights=None):
    """
    使用 CuPy 对三维数组 arr (num_sta, segments, NX)
    沿第二维 (segments) 进行线性叠加。

    参数:
        arr: cp.ndarray
            输入数组，形状为 (num_sta, segments, NX)
        weights: cp.ndarray 或 None
            权重数组，长度为 segments。
            若为 None，则默认等权叠加。

    返回:
        stacked: cp.ndarray
            叠加结果，形状为 (num_sta, NX)
    """
    num_sta, segments, NX = arr.shape

    if weights is None:
        weights = cp.ones(segments, dtype=arr.dtype) / segments
    else:
        weights = cp.asarray(weights, dtype=arr.dtype)
        if len(weights) != segments:
            raise ValueError(
                f"Weights length ({len(weights)}) must equal the number of segments ({segments})"
            )

    # 归一化权重
    weights = weights / cp.sum(weights)

    # 沿第二维加权叠加
    stacked = cp.tensordot(arr, weights, axes=(1, 0))

    return stacked


def pws_time_domain_list(
    arr_list,
    power=2,
    pws_timegate=5.0
):
    """
    arr_list: list of cupy arrays
              arr_list[k].shape = (Nk, NX)
    return  : list of cupy arrays
              out[k].shape = (NX,)
    """

    out = []

    for arr in arr_list:
        # arr: (Nk, NX)
        Nk, NX = arr.shape

        # mask NaN
        mask = ~cp.isnan(arr)
        valid_counts = cp.sum(mask, axis=0)  # (NX,)

        # 防止除零
        valid_counts = cp.maximum(valid_counts, 1)

        # NaN -> 0
        arr0 = cp.nan_to_num(arr)

        # Hilbert transform
        analytic = hilbert(
            arr0,
            axis=-1,
            N=next_fast_len(NX)
        )[:, :NX]

        phase = cp.angle(analytic)

        # Phase stack
        phase_stack = (
            cp.sum(cp.exp(1j * phase) * mask, axis=0)
            / valid_counts
        )

        phase_weight = cp.abs(phase_stack) ** power  # (NX,)

        # Apply PWS
        weighted = arr0 * phase_weight[None, :]

        # Weighted mean
        weighted_sum = cp.sum(weighted * mask, axis=0)
        mean_weighted = weighted_sum / valid_counts

        out.append(mean_weighted)

    return out


def pws_time_domain_new(arr, power=2, pws_timegate=5.):
    num_sta, segments, NX = arr.shape

    # Mask NaN values and count the valid segments
    mask = ~cp.isnan(arr)
    valid_counts = cp.sum(mask, axis=1)

    # Replace NaNs with zeros for the purpose of Hilbert transform
    arr = cp.nan_to_num(arr)

    # Perform Hilbert transform
    analytic = hilbert(arr, axis=-1, N=next_fast_len(NX))[:, :, :NX]
    phase = cp.angle(analytic)

    # Compute phase stack, considering only valid segments
    phase_stack = cp.sum(cp.exp(1j * phase) * mask, axis=1) / valid_counts
    phase_stack = (cp.abs(phase_stack) ** power)[:, cp.newaxis, :]

    # Apply phase stack to original array (with NaNs replaced by zeros)
    weighted = cp.multiply(arr, phase_stack)

    # Compute the weighted mean, considering only valid segments
    weighted_sum = cp.sum(weighted * mask, axis=1)
    mean_weighted = weighted_sum / valid_counts

    return mean_weighted


def pws(fft_arr, power=2, pws_timegate=5.):
    '''
    Performs phase-weighted stack on array of time series. Modified on the noise function by Tim Climents.
    Follows methods of Schimmel and Paulssen, 1997.
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    PARAMETERS:
    ---------------------
    arr: N length array of time series data (numpy.ndarray)
    sampling_rate: sampling rate of time series arr (int)
    power: exponent for phase stack (int)
    pws_timegate: number of seconds to smooth phase stack (float)

    RETURNS:
    ---------------------
    weighted: Phase weighted stack of time series data (numpy.ndarray)
    '''

    num_sta, segmets, NX = fft_arr.shape
    analytic = fft2hilbert(fft_arr, NX)
    phase = cp.angle(analytic)
    phase_stack = cp.mean(np.exp(1j * phase), axis=1)
    phase_stack = (cp.abs(phase_stack) ** (power))[:, cp.newaxis, :]
    # arr = cp.fft.ifftshift(cp.fft.ifft(fft_arr, NX), axes=-1)
    arr = cp.fft.ifft(fft_arr, NX, axis=-1)
    # smoothing
    # timegate_samples = int(pws_timegate * sampling_rate)
    # phase_stack = moving_ave(phase_stack,timegate_samples)
    weighted = cp.multiply(arr, phase_stack)
    return cp.real(cp.fft.ifftshift(cp.mean(weighted, axis=1), axes=-1))


def fft2hilbert(x, N):
    Xf = x
    axis = -1
    h = cp.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    output = cp.fft.ifft(Xf * h, axis=axis)
    return output


def auto_chunk(shape):
    """智能分块策略：取各维度10%大小，最小为1"""
    return tuple(max(1, s // 10) for s in shape)


def store_param(group, key, value):
    """存储参数到 HDF5 属性，处理特殊类型"""
    # 处理 ruamel.yaml 的特殊类型
    if hasattr(value, 'anchor') or hasattr(value, 'value'):
        value = str(value)

    # 处理普通字符串
    if isinstance(value, str):
        value = str(value)  # 确保是普通 Python 字符串

    # 尝试存储为 NumPy 数组
    try:
        group.attrs[key] = np.array(value)
    except TypeError:
        # 如果失败，存储为字符串
        try:
            group.attrs[key] = str(value)
        except Exception as e:
            print(f"Unable to store parameter {key}={value} ({type(value)}): {str(e)}")
            group.attrs[key] = "ERROR_STORING_VALUE"
    # """类型安全存储参数"""
    # if isinstance(value, (str, bytes)):
    #     group.attrs[key] = value
    # elif isinstance(value, (int, float)):
    #     group.attrs[key] = value
    # elif isinstance(value, (list, tuple)):
    #     group.attrs[key] = np.array(value)
    # else:
    #     group.attrs[key] = str(value)


def load_params(param_group):
    """安全加载参数并转换回原生类型"""
    params = {}
    for k in param_group.attrs:
        val = param_group.attrs[k]
        if isinstance(val, bytes):
            params[k] = val.decode('utf-8')
        elif isinstance(val, np.ndarray):
            params[k] = val.tolist()
        elif isinstance(val, np.generic):
            params[k] = val.item()
        else:
            params[k] = val
    return params


def _load_params_safe(group):
    """安全参数加载"""
    params = {}
    for key in group.attrs:
        # 基础类型直接读取
        val = group.attrs[key]
        # 处理特殊字符串
        if isinstance(val, bytes):
            val = val.decode()
        if val == "None":
            params[key] = None
        elif isinstance(val, str) and val.startswith('datetime:'):
            params[key] = datetime.fromisoformat(val[9:])
        else:
            params[key] = val

    # 处理子组（字典）
    for key in group:
        subgroup = group[key]
        params[key] = _load_params_safe(subgroup)

    return params


def write_single_h5(filename, data, params=None, compression="gzip"):
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('data', data=data, compression=compression)
        # 存储 dtype 的标准名称（关键修改）
        dset.attrs['dtype'] = data.dtype.name  # 例如 'float32'
        if params:
            param_group = f.create_group('params')
            for k, v in params.items():
                store_param(param_group, k, v)


def read_single_h5(filename):
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]
        # 使用 np.dtype 安全转换（关键修改）
        dtype = np.dtype(f['data'].attrs['dtype'])
        data = data.astype(dtype)
        params = {}
        if 'params' in f:
            params = load_params(f['params'])

        return data, params


def res_write(corr_all_stack, params):
    tr = Trace()
    st = Stream()
    # corr_all_stack = np.array(corr_all_stack.get())
    corr_all_stack = np.asarray(corr_all_stack)
    if corr_all_stack.ndim == 4 and corr_all_stack.shape[-2] == 1:
        corr_all_stack = np.squeeze(corr_all_stack, axis=-2)
    if corr_all_stack.ndim == 2:
        corr_all_stack = corr_all_stack[np.newaxis, ...]
    # data_temp = np.zeros_like(corr_all_stack)
    # dlen = int(cal_params['c_range'] / 2)
    # data_temp[:, :dlen, :] = corr_all_stack[:, dlen + 1:, :]
    # data_temp[:, dlen:, :] = corr_all_stack[:, :dlen + 1, :]
    # corr_all_stack = data_temp
    tr.stats.npts = len(corr_all_stack[0, 0])
    tr.stats.delta = "%0.3f" % (params['delta'])
    tr.stats.network = "ustc"
    # tr.stats.sampling_rate = cal_params['samplerate']/2
    # tr.stats.station = str(params[1]) + str(params[1])

    st.append(tr)
    if params['meta_flag']:
        with open(params['meta_file'], 'r') as file:
            stas = file.readlines()
        tra_meta = [sta.split(' ')[0] for sta in stas]
        lat_meta = [float(sta.split(' ')[1]) for sta in stas]
        lon_meta = [float(sta.split(' ')[2]) for sta in stas]

    # for sta in range(len(corr_all_stack)):
    def sac_write(sta):
        outpath = params['outputdir'] + '/NCF2/S' + str(sta * params['step']).zfill(4) + '/'
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        if params['whole_array']:
            for tra in range(len(corr_all_stack[0])):
                if tra == sta * params['step']:
                    continue
                st[0].data = corr_all_stack[sta][tra]
                sac = SACTrace.from_obspy_trace(tr)
                if params['meta_flag'] and not lat_meta[sta * params['step']]:
                    sac.stla = float(lat_meta[sta * params['step']])
                    sac.stlo = float(lon_meta[sta * params['step']])
                    sac.evla = float(lat_meta[tra])
                    sac.evlo = float(lon_meta[tra])
                else:
                    # sac.evlo = tr_source[0].stats.sac.stlo
                    # sac.stlo = tr_receiver[0].stats.sac.stlo
                    # sac.evla = tr_source[0].stats.sac.stla
                    # sac.stla = tr_receiver[0].stats.sac.stla
                    sac.evlo = '-123456'
                    sac.stlo = '-123456'
                    sac.evla = '-123456'
                    sac.stla = '-123456'
                sac.b = 0
                if sac.stla is None or sac.stla == -123456:
                    dist = params['dr'] * tra
                else:
                    dist, az, baz = gps2dist_azimuth(sac.evla, sac.evlo, sac.stla, sac.stlo)
                # dist = params['dr']
                ccf_name = 'COR_' + str("%04d" % (sta * params['step'])) + '_' + str(
                    "%04d" % tra) \
                           + '_' + str("%08.4f" % dist) + '.sac'
                sac.write(outpath + ccf_name, byteorder='little')
        else:
            for tra in range(1, len(corr_all_stack[0])):
                st[0].data = corr_all_stack[sta][tra]
                sac = SACTrace.from_obspy_trace(tr)
                if params['meta_flag'] and not lat_meta[sta * params['step']]:
                    sac.stla = float(lat_meta[sta * params['step']])
                    sac.stlo = float(lon_meta[sta * params['step']])
                    sac.evla = float(lat_meta[sta * params['step'] + tra])
                    sac.evlo = float(lon_meta[sta * params['step'] + tra])
                else:
                    # sac.evlo = tr_source[0].stats.sac.stlo
                    # sac.stlo = tr_receiver[0].stats.sac.stlo
                    # sac.evla = tr_source[0].stats.sac.stla
                    # sac.stla = tr_receiver[0].stats.sac.stla
                    sac.evlo = '-123456'
                    sac.stlo = '-123456'
                    sac.evla = '-123456'
                    sac.stla = '-123456'
                sac.b = 0
                if sac.stla is None or sac.stla == -123456:
                    dist = params['dr'] * tra
                else:
                    dist, az, baz = gps2dist_azimuth(sac.evla, sac.evlo, sac.stla, sac.stlo)
                # dist = params['dr']
                ccf_name = 'COR_' + str("%04d" % (sta * params['step'])) + '_' + str(
                    "%04d" % (sta * params['step'] + tra)) \
                           + '_' + str("%08.4f" % dist) + '.sac'
                sac.write(outpath + ccf_name, byteorder='little')

    with ThreadPoolExecutor() as executor:
        executor.map(sac_write, range(len(corr_all_stack)))


def acorr(r_data, Nfft2):
    source_white = cp.conjugate(r_data)
    corr = source_white @ r_data
    corr[:, :, :Nfft2] = corr[:, :, :Nfft2] - cp.mean(corr[:, :, :Nfft2], axis=-1)[:, :, cp.newaxis]
    corr[:, :, -(Nfft2) + 1:] = cp.flip(cp.conj(corr[:, :, 1:(Nfft2)]), axis=-1)
    corr[:, :, Nfft2] = complex(0, 0)
    corr[:, :, 0] = complex(0, 0)
    s_corr = cp.real(cp.fft.ifftshift(cp.fft.ifft(corr, Nfft2 * 2, axis=-1), axes=-1))

    return s_corr


def finddatastruct(file, prefix='', data=None):
    """Return the most likely data dataset path in an H5 file."""
    data_struct = find_data_struct(file)
    if data_struct is None:
        raise KeyError("No suitable numeric 2D dataset found in H5 file")
    return data_struct


def numpy2tensor(a):
    """
        transform numpy data into tensor
    """
    if not torch.is_tensor(a):
        return torch.tensor(a).to(torch.float32)
    else:
        return a.to(torch.float32)


def tensor2numpy(a):
    """
        transform tensor data into numpy
    """
    if not torch.is_tensor(a):
        return a
    else:
        return a.detach().numpy()


def list2numpy(a):
    """
        transform numpy data into tensor
    """
    if isinstance(a, list):
        return np.array(a)
    else:
        return a


def numpy2list(a):
    """
        transform numpy data into tensor
    """
    if not isinstance(a, list):
        return a.tolist()
    else:
        return a


def next_pow_2(i):
    """
    Find the next power of two >= i

    >>> next_pow_2(5)
    8
    >>> next_pow_2(250)
    256
    """
    if i <= 0:
        return 1
    return 1 << (i - 1).bit_length()


def beamforming_analysis(data, x, y, fs, win_len=1500, n_windows=2,
                         azimuth_step=5, slowness_min=0.05, slowness_max=0.5, slowness_points=40):
    """
    对已有阵列数据做beamforming，并绘制多个时间窗结果。

    参数:
        data: np.ndarray, shape (n_stations, npts)，阵列数据
        x, y: np.ndarray, 阵列传感器位置坐标（单位：米）
        fs: float，采样率（Hz）
        win_len: int，单个时间窗长度（采样点数）
        n_windows: int，时间窗数量（均匀选取）
        azimuth_step: float，扫描方位角步长（度）
        slowness_min: float，扫描最小走时倒数（s/km）
        slowness_max: float，扫描最大走时倒数（s/km）
        slowness_points: int，走时倒数采样点数

    返回:
        detected_azimuths: list，时间窗对应的检测方位角列表
    """

    n_stations, npts = data.shape
    azimuths = np.arange(0, 360, azimuth_step)
    slownesses = np.linspace(slowness_min, slowness_max, slowness_points)

    def beamform_window(data_win):
        n_stations_win, npts_win = data_win.shape
        nfft = next_pow_2(npts_win)
        freqs = rfftfreq(nfft, 1 / fs)
        data_f = rfft(data_win, nfft, axis=1)

        beam_power = np.zeros((len(slownesses), len(azimuths)))
        for i, slow in enumerate(slownesses):
            for j, az in enumerate(azimuths):
                delays = slow * (x * np.cos(np.deg2rad(az)) + y * np.sin(np.deg2rad(az))) / 1000.0
                phase = np.exp(-2j * np.pi * np.outer(freqs, delays))
                beam = np.abs(np.sum(data_f * phase.T, axis=0))
                beam_power[i, j] = np.mean(beam)
        return beam_power

    # 计算时间窗起点，均匀选取n_windows个时间窗
    start_indices = np.linspace(0, npts - win_len, n_windows, dtype=int)

    detected_azimuths = []

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), subplot_kw={'projection': 'polar'})

    for k, start in enumerate(start_indices):
        end = start + win_len
        window_data = data[:, start:end]
        power = beamform_window(window_data)
        power_db = 10 * np.log10(power / np.max(power))

        max_idx = np.unravel_index(np.argmax(power), power.shape)
        det_az = azimuths[max_idx[1]]
        detected_azimuths.append(det_az)

        ax = axes.flat[k]
        im = ax.contourf(np.deg2rad(azimuths), slownesses, power_db, levels=40, cmap='viridis')
        ax.plot(np.deg2rad(det_az), slownesses[max_idx[0]], 'r*', markersize=12, label='Detected')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Window {k + 1}: Detected Azimuth = {det_az:.1f}°")
        if k == 0:
            ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.3))

    plt.tight_layout()
    plt.show()

    return detected_azimuths


def beamforming_analysis_gpu(data, x, y, fs,
                             win_len=1500, n_windows=2,
                             azimuth_step=5,
                             slowness_min=0.05, slowness_max=0.5, slowness_points=40,
                             ):
    """
    GPU 加速的 beamforming（CuPy 实现，推荐用于 n_stations x npts 的 data）
    - data: (n_stations, npts) numpy or cupy array (会转为 cupy)
    - x, y: (n_stations,) 坐标，单位米
    - fs: sampling rate (float)
    - win_len: 单个时窗长度（采样点）
    - n_windows: 要计算的时窗数（均匀取）
    - azimuth_step: 方位扫描步长（度）
    - slowness_min/slowness_max/slowness_points: slowness 扫描范围与点数 (s/km)
    - plot: 是否绘图（True 会把结果转为 numpy 并用 matplotlib 绘图）
    返回:
      detected_azimuths: 列表，每个时窗检测到的方位角（度）
    """

    # --------- 将输入转为 CuPy ---------
    data_gpu = cp.asarray(data)  # (n_stations, npts)
    x_gpu = cp.asarray(x).astype(cp.float32)
    y_gpu = cp.asarray(y).astype(cp.float32)
    fs_f = float(fs)
    n_stations, npts = data_gpu.shape

    # 参数数组（CuPy）
    azimuths_cpu = np.arange(0, 360, azimuth_step)  # small, keep CPU for titles etc.
    azimuths = cp.asarray(azimuths_cpu.astype(np.float32))  # shape (n_az,)
    az_rad = cp.deg2rad(azimuths)  # (n_az,)

    slownesses_cpu = np.linspace(slowness_min, slowness_max, slowness_points).astype(np.float32)
    slownesses = cp.asarray(slownesses_cpu)  # (n_slow,)

    n_az = azimuths.size
    n_slow = slownesses.size

    # 预计算站点在每个方位的投影： proj[sta, az] = x[sta]*cos(az) + y[sta]*sin(az)
    # 维度： (n_stations, n_az)
    proj = x_gpu[:, None] * cp.cos(az_rad)[None, :] + y_gpu[:, None] * cp.sin(az_rad)[None, :]

    # 时间窗起点（在 CPU 上）
    if n_windows <= 1:
        start_indices = np.array([0], dtype=int)
    else:
        start_indices = np.linspace(0, npts - win_len, n_windows, dtype=int)

    detected_azimuths = []
    all_beam_power = []
    # 主循环：每个时间窗
    for win_idx, start in enumerate(start_indices):
        end = int(start + win_len)
        if end > npts:
            end = npts
            start = end - win_len
        # 保持为 cupy 数组（不要 .get()）
        window_data_gpu = data_gpu[:, start:end]  # shape (n_stations, win_len)

        # FFT on GPU
        nfft = next_pow_2(window_data_gpu.shape[1])
        freqs = cp.fft.rfftfreq(nfft, d=1.0 / fs_f)  # cupy array (nfreq,)
        data_f = cp.fft.rfft(window_data_gpu, n=nfft, axis=1)  # shape (n_stations, nfreq), complex64/128

        nfreq = freqs.size

        # 为每个 slowness 计算 beam_power（loop 在 slowness 维，通常 slowness_points 稍小）
        # 但每次循环都在 GPU 上做频率/站点/方位的批量运算，避免 Python 层的 azimuth 循环
        beam_power_gpu = cp.zeros((n_slow, n_az), dtype=cp.float32)

        # 将 data_f 转置为 (nfreq, n_stations) 方便后续广播
        data_f_tf = data_f.T  # (nfreq, n_stations), complex

        # 为数值稳定，若数据为 complex128，可根据显存与精度改成 complex64
        # 开始对每个 slowness 批处理
        for si in range(n_slow):
            slow = slownesses[si]  # scalar cupy
            # delays (station, az) in seconds: proj / 1000 * slow
            delays_sta_az = (proj * slow) / 1000.0  # (n_stations, n_az)

            # 计算相位： shape -> (nfreq, n_stations, n_az)
            # freqs[:,None,None] * delays[None,:,:] -> (nfreq, n_stations, n_az)
            phase = cp.exp(-2j * cp.pi * freqs[:, None, None] * delays_sta_az[None, :, :])

            # data_f_tf shape (nfreq, n_stations) -> expand to (nfreq, n_stations, 1)
            data_exp = data_f_tf[:, :, None]  # (nfreq, n_stations, 1)

            # multiply并在 station 轴求和 -> result (nfreq, n_az)
            summed = cp.sum(data_exp * phase, axis=1)  # (nfreq, n_az), complex

            # 取幅值并对频率轴做平均 -> (n_az,)
            summed_abs = cp.abs(summed)  # (nfreq, n_az)
            power_az = cp.mean(summed_abs, axis=0)  # (n_az,)

            beam_power_gpu[si, :] = power_az.astype(cp.float32)

            # 可选：释放临时内存（交给 GC/内存池）
            del delays_sta_az, phase, data_exp, summed, summed_abs, power_az

        # 找最大
        max_idx = cp.unravel_index(cp.argmax(beam_power_gpu), beam_power_gpu.shape)
        det_slow_idx, det_az_idx = int(max_idx[0].item()), int(max_idx[1].item())
        det_az = float(azimuths_cpu[det_az_idx])
        detected_azimuths.append(det_az)

        # === 关键：把数据拿到 CPU 返回给绘图函数 ===
        all_beam_power.append(cp.asnumpy(beam_power_gpu))

        del window_data_gpu, data_f, data_f_tf, beam_power_gpu

    return detected_azimuths, all_beam_power, slownesses_cpu, azimuths_cpu, det_slow_idx


def plot_beamforming_results(all_beam_power, slownesses, azimuths, det_slow_idx, detected_azimuths=None, save=0,
                             nplot_max=9):
    """
    all_beam_power: list of 2D numpy arrays, each shape (n_slow, n_az)
    slownesses: 1D numpy array length n_slow
    azimuths: 1D numpy array length n_az (degrees)
    detected_azimuths: optional list (not used for marker placement here)
    """
    n_windows = len(all_beam_power)
    nplot = min(n_windows, nplot_max)
    nrows = int(np.ceil(np.sqrt(nplot)))
    ncols = int(np.ceil(nplot / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), subplot_kw={'projection': 'polar'})
    axes = np.array(axes).reshape(-1)

    az_rad = np.deg2rad(azimuths)  # radian vector for contourf

    for i in range(nplot):
        ax = axes[i]
        power = all_beam_power[i]  # shape (n_slow, n_az)
        # safety: ensure shape matches
        power = np.asarray(power)
        if power.ndim != 2:
            raise ValueError(f"power must be 2D array, got shape {power.shape}")

        # convert to dB for plotting
        power_db = 10 * np.log10(power / np.max(power))

        det_slow = slownesses[det_slow_idx]

        # contourf expects (theta, r, Z) with Z shape (len(r), len(theta))
        cs = ax.contourf(az_rad, slownesses, power_db, levels=40, cmap='viridis')

        # plot detected point at correct (theta, r)
        ax.plot(np.deg2rad(detected_azimuths[i]), det_slow, 'r*', markersize=12,
                label=f'Det: {detected_azimuths[i]:.1f}°, s={det_slow:.3f}')

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Window {i + 1}: Det={detected_azimuths[i]:.1f}° / s={det_slow:.3f}")
        ax.legend(loc='lower left')

    # hide any unused axes
    for j in range(nplot, len(axes)):
        axes[j].axis('off')

    fig.colorbar(cs, ax=axes[:nplot].tolist(), orientation='vertical', fraction=0.03)
    fig.tight_layout()
    if save:
        if isinstance(save, (str, os.PathLike)):
            save_path = os.fspath(save)
        else:
            save_path = os.path.join("output", "beamforming.png")
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def load_and_project(filename):
    if Transformer is None:
        raise RuntimeError(
            "Coordinate projection requires the optional 'pyproj' package. "
            "Install it when meta_flag is enabled."
        )
    lats = []
    lons = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                # 如果格式不对，依然补0
                lats.append(0.0)
                lons.append(0.0)
                continue
            sta, lat_str, lon_str = parts
            try:
                lat, lon = float(lat_str), float(lon_str)
            except:
                lat, lon = 0.0, 0.0
            if lon == -12345 and lat == -12345:
                # 无效点，补0
                lat, lon = 0.0, 0.0
            # 简单范围检查，不合规填0
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                print(f"Warning: invalid coord {lat}, {lon}, replaced with 0")
                lat, lon = 0.0, 0.0

            lats.append(lat)
            lons.append(lon)

    print(f"Points read: {len(lons)}")

    # 投影之前先转换成 numpy array
    lats = np.array(lats)
    lons = np.array(lons)

    # 投影无效点经纬度为0，转换后对应的平面坐标也就是投影0点
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x, y = transformer.transform(lons, lats)  # lon, lat顺序

    x = np.array(x)
    y = np.array(y)
    return x, y


def read_tdms_concat_auto(tdms_path, fill_value=0.0):
    if TdmsFile is None:
        raise RuntimeError(
            "TDMS support requires the optional 'nptdms' package. "
            "Install it before processing TDMS files."
        )
    tdms = TdmsFile.read(tdms_path)

    # ---------- 1. 找有效 group ----------
    valid_groups = [g for g in tdms.groups() if len(g.channels()) > 0]
    if not valid_groups:
        raise RuntimeError("No TDMS group containing data was found")

    # ---------- 2. 用第一个 group 定义 channel 集合 ----------
    ref_group = valid_groups[0]
    ch_names = [ch.name for ch in ref_group.channels()]
    nch = len(ch_names)
    nt_min = len(ref_group.channels()[0])

    # ---------- 3. 预分配 ----------
    nt_total = nt_min * len(valid_groups)
    data = np.full((nch, nt_total), fill_value, dtype=np.float32)

    # ---------- 4. 按 channel 名对齐拼接 ----------
    t_index = 0
    for g in valid_groups:
        print(g.name, len(g.channels()))

        t0 = t_index * nt_min
        t1 = t0 + nt_min

        ch_dict = {ch.name: ch for ch in g.channels()}

        for ich, name in enumerate(ch_names):
            if name in ch_dict:
                data[ich, t0:t1] = ch_dict[name][:]
            else:
                # 缺失 channel：保持 fill_value
                pass

        t_index += 1

    return data
