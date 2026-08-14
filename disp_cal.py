# -*- coding: utf-8 -*-
# @Time : 2024/4/17 10:58
# @Site : 
# @File : disp_cal.py
# @Software: PyCharm
import math

import numpy as np
from numba import cuda
from scipy import fft
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, rfftfreq
from scipy.signal import butter, filtfilt
from scipy.special import j0
# import ccfj


def _ensure_cc_cube(cc_data):
    """Return CCF data as (source, receiver, time), preserving one source."""
    cc = np.asarray(cc_data).squeeze()
    if cc.ndim == 2:
        cc = cc[np.newaxis, ...]
    if cc.ndim != 3:
        raise ValueError(
            f"CCF data must have shape (source, receiver, time), got {cc.shape}"
        )
    return cc


def butter_lowpass(cutoff, fs, order=5):
    """
    参数：
        cutoff : 截止频率（Hz）
        fs     : 采样频率（Hz）
        order  : 滤波器阶数（默认5阶）
    """
    nyq = 0.5 * fs          # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def time_to_freq(ncfst, channels, step=1, do_ifftshift=True):
    cc = ncfst[channels[0]:channels[1], :].copy()
    cc = cc[::step, :]
    n_rows, n_time_points = cc.shape

    n_cols_freq = n_time_points // 2 + 1
    freq_matrix = np.zeros((n_rows, n_cols_freq), dtype=complex)


    for i in range(n_rows):
        row = cc[i, :]
        if do_ifftshift:
            row = np.fft.ifftshift(row)
        freq_matrix[i, :] = np.fft.rfft(row)

    return freq_matrix


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def fk_filter(data_filter, dt, dx, w=0.5, show=True, freq_window=1):
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
    fk = fk * mask
    if show:
        freq_window = freq_window/dt
        fk_1 = fk.copy()
        pmin = -10
        fs = 1 / dt
        lf = len(f)
        P = abs(np.fft.fftshift(fk_1))
        P /= P.max()
        P = 10 * np.log10(P)
        P2 = abs(fk)
        P2 /= P2.max()
        # P2 = 10 * np.log10(P2)
        plt.pcolormesh(k, f[int(lf / 2):int(lf / 2 + freq_window)],
                       P[:, int(lf / 2):int(lf / 2 + freq_window)].T, cmap='viridis', vmin=pmin, vmax=0)
        plt.xlabel('Wavenumber')
        plt.ylabel('Frequancy(Hz)')
        plt.title('F-K')
        plt.show()
    PP = np.fft.ifft2(fk)
    return PP.real


@cuda.jit
def Phase_cal(f_ind, nx, x, rec_fft_n, E, fmin_ind, f, v, dx=8.18):
    i, j = cuda.grid(2)
    offset = 0
    if i < E.shape[0] and j < E.shape[1]:
        f_ind[i, j] = fmin_ind + j
        for k in range(nx):
            x[i, j] = offset + k * dx
            E[i, j] = E[i, j] + math.e ** (1j * 2 * math.pi * f[f_ind[i, j]] * x[i, j] / v[i]) * rec_fft_n[
                k, f_ind[i, j]]

@cuda.jit
def Phase_cal(f_ind, nx, x_dev, rec_fft_n, E_dev, fmin_ind, f_dev, v_dev, dx):
    i, j = cuda.grid(2)
    # E_dev shape: (lv, lf)
    if i < E_dev.shape[0] and j < E_dev.shape[1]:
        fi = fmin_ind + j
        f_ind[i, j] = fi
        # accumulate real and imag separately
        sum_re = 0.0
        sum_im = 0.0
        for k in range(nx):
            xpos = k * dx
            x_dev[i, j] = xpos
            theta = 2.0 * math.pi * f_dev[fi] * xpos / v_dev[i]
            c = math.cos(theta)
            s = math.sin(theta)
            val = rec_fft_n[k, fi]
            val_re = val.real
            val_im = val.imag
            prod_re = c * val_re - s * val_im
            prod_im = c * val_im + s * val_re
            sum_re += prod_re
            sum_im += prod_im
        E_dev[i, j] = complex(sum_re, sum_im)


def cal(params, cc_t, callback_func, callback_plotdata, callback_str1):
    start_sta = int(np.ceil(params['range'][0]/params['step']))
    end_sta = int(np.floor(params['range'][1]/params['step']))
    cc_t = _ensure_cc_cube(cc_t)
    cc = cc_t[:, 1:, :]
    # files.sort(key=lambda files: files.split('_')[-1])
    fmin = params['fmin']
    fmax = params['fmax']
    vmax = params['vmax']
    vmin = params['vmin']
    dv = params['dv']
    TBP_x = 8
    TBP_y = 8
    v = np.arange(vmax, vmin, -dv)
    lv = len(v)
    dt = params['dt']
    dr = params['dr']
    E_all = []
    # cc = cc.decimate(factor=10)
    len_t = int((params['cc_len'] / dt) / 2)
    # cc = cc[:, :, len_t - int(params['time_range'] / dt):len_t + int(params['time_range'] / dt)]

    [nx, nt1] = cc[0, :, len_t - int(params['time_range'] / dt):len_t + int(params['time_range'] / dt)].shape
    Fs = 1 / dt
    r_scale = [tra * params['dr'] for tra in range(cc.shape[-2])]
    t_scalef = [i * dt for i in list(range(-int(params['time_range'] / dt), int(params['time_range'] / dt)))]
    f = Fs * np.arange(0, (nt1/ 2)) / nt1
    ind = [i for i, num in enumerate(f) if num > fmin]
    fmin_ind = ind[0] - 1
    ind = [i for i, num in enumerate(f) if num >= fmax]
    fmax_ind = ind[0]
    freq = f[fmin_ind:fmax_ind]
    period = 1 / freq
    lf = len(freq)
    threadperblock = (TBP_x, TBP_y)
    blockspergrid_x = int(np.ceil(lv / threadperblock[0]))
    blockspergrid_y = int(np.ceil(lf / threadperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    for i in range(0, end_sta-start_sta):
        callback_str1('Calculating disp')
        callback_func(i+start_sta)
        seisdata_full = np.zeros((cc.shape[1], cc.shape[-1]))
        for i_sta in range(cc.shape[1]):
            seisdata_full[i_sta, :] = cc_t[i, i_sta, :]
            # seisdata_full[i_sta] /= seisdata_full[i_sta].max()
        # # seisdata_full[i_sta, int(time_len * sampling) - 4:int(time_len * sampling) + 4] = 0
        #     if len(params['cutpoint']) > 1:
        #         seisdata_full[i_sta,
        #         min(int((len_t + (params['cutpoint'][1][i_sta]) / dt)), seisdata_full.shape[-1]):] = 0
        #         seisdata_full[i_sta,
        #         :max(int((len_t + (params['cutpoint'][0][i_sta]) / dt)), 0)] = 0
        seisdata_full_t = seisdata_full[:,
                        int(len_t) - int(params['time_range'] / params['dt']):int(len_t) + int(
                            params['time_range'] / params['dt'])]
        # seisdata_full_t[:, int(params['time_range'] / params['dt']) - 15:int(params['time_range'] / params['dt']) + 15] = 0
            # causal
        if params['Part'] == 'causal':
            seisdata = seisdata_full_t[:, int(params['time_range'] / params['dt']):]
        # print(i_sta)
        # seisdata[i_sta, :8] = 0
        # seisdata[i_sta, i_sta*4:] = 0
        if params['Part'] == 'noncausal':
            seisdata = seisdata_full_t[:, :int(params['time_range'] / params['dt'])]
            seisdata = seisdata[:, ::-1]
        # seisdata[i_sta, :32] = 0
        seisdata = np.ascontiguousarray(seisdata)
        n_cc = seisdata
        # cc = cc.normalize()
        # plt.title('NCF_%s' %files[i].split('/')[-1] )
        [nx, nt] = n_cc.shape
        f = Fs * np.arange(0, (nt / 2)) / nt

        fmin_ind = np.searchsorted(f, fmin) - 1
        fmax_ind = np.searchsorted(f, fmax)

        freq = f[fmin_ind:fmax_ind]
        lf = len(freq)
        Fs = 1 / dt
        f = Fs * np.arange(0, (nt / 2)) / nt
        rec_fft = np.zeros((nx, nt), dtype=complex)
        for j in range(nx):
            rec_fft[j, :] = fft.fft(n_cc[j, :], nt, 0)
        rec_fft_amp = abs(rec_fft)
        rec_fft_n = rec_fft / rec_fft_amp
        rec_fft_n = cuda.to_device(rec_fft_n)
        E = cuda.to_device(np.zeros((lv, lf), dtype=complex))
        x = cuda.to_device(np.zeros((lv, lf), dtype=int))
        f_ind = cuda.to_device(np.zeros((lv, lf), dtype=int))
        temp = cuda.to_device(np.zeros((1, 1), dtype=complex))
        Phase_cal[blockspergrid, threadperblock](f_ind, nx, x, rec_fft_n, E, fmin_ind, f, v, dr)
        E = E.copy_to_host()
        for j in range(E.shape[1]):
            E[:, j] = abs(E[:, j] / max(abs(E[:, j])))
        extent = [freq[0], freq[-1], vmin, vmax]
        # callback_plotdata(E)
        E_all.append(E.real[::-1, :])
        # sor_name = files[i].split('S')[1]
        # plt.figure()
        # plt.imshow(np.real(E), extent=extent, aspect='auto', interpolation='bicubic', cmap='jet')
        # # 去掉坐标轴和边框
        # plt.axis('off')  # 去掉坐标轴
        # # 调整图像的边缘和边距
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # # 显示图像
        # plt.show()

    callback_plotdata([np.array(E_all), extent])


# def cc_fj(params,cc_t, callback_func, callback_plotdata, callback_str1):
#     cc_t = cc_t.squeeze()
#     ch1 = 1
#     ch2 = cc_t.shape[1]
#     cc = cc_t[:, :, :]
#     start_sta = int(np.ceil(params['range'][0]/params['step']))
#     end_sta = int(np.floor(params['range'][1]/params['step']))
#     fmax = int(params['fmax'])
#     fmin = params['fmin']
#     vmax = params['vmax']
#     vmin = params['vmin']
#     dv = params['dv']
#     dt = params['dt']
#     len_t = int((params['cc_len'] / dt) / 2)
#     # r_scale = [tra * params['dr'] for tra in range(1, cc_t.shape[-2])]
#     r_scale = (np.arange(cc_t.shape[-2])) * params['dr']
#     r_scale = r_scale[ch1:ch2]
#     c_scale = np.round(np.linspace(vmin, vmax, int((vmax - vmin) / dv) + 1))
#     f_scale  = np.fft.rfftfreq(cc_t.shape[-1], dt)
#     ind = [i for i, num in enumerate(f_scale) if num > fmin]
#     fmin_ind = ind[0] - 1
#     ind = [i for i, num in enumerate(f_scale) if num >= fmax]
#     fmax_ind = ind[0]
#     freq = f_scale[fmin_ind:fmax_ind]
#     ds_all=[]
#     for i in range(0, end_sta - start_sta):
#         callback_str1('Calculating disp')
#         callback_func(i+start_sta)
#         seisdata_full = np.zeros((cc.shape[1], cc.shape[-1]))
#         for i_sta in range(cc.shape[1]):
#             seisdata_full[i_sta, :] = cc_t[i, i_sta, :]
#             # seisdata_full[i_sta] /= seisdata_full[i_sta].max()
#         seisdata_full_t = seisdata_full[:,
#                         int(len_t) - int(params['time_range'] / params['dt']):int(len_t) + int(
#                             params['time_range'] / params['dt'])]
#         step = 1
#         seisdata_fj = time_to_freq(seisdata_full,  channels=[ch1, ch2], step=step, do_ifftshift=True)
#
#         ds00 = ccfj.fj_noise(np.real(seisdata_fj), r_scale, c_scale, f_scale, fstride=1, itype=0, func=0)
#         # ds01 = ccfj.fj_noise(np.real(seisdata_fj), r_scale, c_scale, f_scale, fstride=1, itype=1, func=0)
#         # ds10 = ccfj.fj_noise(np.real(seisdata_fj), r_scale, c_scale, f_scale, fstride=1, itype=0, func=1)
#         # ds11 = ccfj.fj_noise(np.real(seisdata_fj), r_scale, c_scale, f_scale, fstride=1, itype=1, func=1)
#         ds_all.append(ds00)
#     extent = [freq[0], freq[-1], vmin, vmax]
#     callback_plotdata([np.array(ds_all), extent])


def _dispersion_input(params, cc_t):
    """Prepare source gathers and axes shared by the F-J implementations."""
    cc = _ensure_cc_cube(cc_t)
    if cc.shape[1] < 3:
        raise ValueError("CCF data must have shape (source, trace, time)")

    dt = float(params['dt'])
    dr = float(params['dr'])
    if dt <= 0 or dr <= 0:
        raise ValueError("dt and dr must be positive")

    step = max(float(params.get('step', 1)), 1.0)
    range_value = params.get('range', [0, cc.shape[0] * step])
    start = max(int(np.ceil(float(range_value[0]) / step)), 0)
    end = min(int(np.floor(float(range_value[1]) / step)), cc.shape[0])
    if end <= start:
        start, end = 0, cc.shape[0]

    center = cc.shape[-1] // 2
    requested = max(int(round(float(params['time_range']) / dt)), 2)
    side_samples = min(requested, center, cc.shape[-1] - center)
    if side_samples < 4:
        raise ValueError("The selected CCF time window is too short")

    part = str(params.get('Part', 'causal')).lower()
    gathers = []
    for source_index in range(start, end):
        source = cc[source_index, 1:, :]
        if part == 'causal':
            branch = source[:, center:center + side_samples]
        elif part == 'noncausal':
            branch = source[:, center - side_samples:center][:, ::-1]
        else:
            raise ValueError("Part must be 'causal' or 'noncausal'")
        gathers.append(np.ascontiguousarray(branch, dtype=np.float64))

    frequencies_all = np.fft.rfftfreq(side_samples, dt)
    use_frequency = ((frequencies_all >= float(params['fmin'])) &
                     (frequencies_all <= float(params['fmax'])))
    frequencies = frequencies_all[use_frequency]
    if frequencies.size == 0:
        raise ValueError("No FFT frequencies fall inside fmin/fmax")

    velocities = np.arange(
        float(params['vmin']),
        float(params['vmax']) + 0.5 * float(params['dv']),
        float(params['dv']),
    )
    if velocities.size == 0 or np.any(velocities <= 0):
        raise ValueError("The velocity range must contain positive values")
    offsets = np.arange(1, cc.shape[1], dtype=np.float64) * dr
    return gathers, use_frequency, frequencies, velocities, offsets, start


def _frequency_spectra(gathers, use_frequency):
    spectra = []
    for gather in gathers:
        taper = np.hanning(gather.shape[-1])
        transformed = np.fft.rfft(gather * taper, axis=-1)[:, use_frequency]
        spectra.append(transformed)
    return np.asarray(spectra)


def _normalize_frequency_columns(image):
    scale = np.max(np.abs(image), axis=0, keepdims=True)
    return np.divide(image, scale, out=np.zeros_like(image), where=scale > 0)


def cc_fj(params, cc_t, callback_func, callback_plotdata, callback_str1):
    """Frequency-Bessel (F-J) transform implemented with SciPy."""
    (gathers, use_frequency, frequencies, velocities, offsets,
     progress_start) = _dispersion_input(params, cc_t)
    spectra = _frequency_spectra(gathers, use_frequency)
    radial_weights = offsets * np.gradient(offsets)
    result = np.empty((len(gathers), velocities.size, frequencies.size), dtype=np.float64)

    callback_str1('Calculating CC-FJ dispersion')
    for frequency_index, frequency in enumerate(frequencies):
        argument = (2.0 * np.pi * frequency * offsets[None, :] /
                    velocities[:, None])
        kernel = j0(argument) * radial_weights
        result[:, :, frequency_index] = np.abs(np.einsum(
            'vr,sr->sv', kernel, spectra[:, :, frequency_index], optimize=True
        ))
        callback_func(progress_start + int(
            (frequency_index + 1) * len(gathers) / frequencies.size
        ))

    for source_index in range(result.shape[0]):
        result[source_index] = _normalize_frequency_columns(result[source_index])
    extent = [float(frequencies[0]), float(frequencies[-1]),
              float(velocities[0]), float(velocities[-1])]
    callback_plotdata([result, extent])


@cuda.jit(device=True)
def bilinear_interp(x, y, x_grid, y_grid, data):
    nx = x_grid.shape[0]
    ny = y_grid.shape[0]
    i = 0
    while i < nx - 2 and x > x_grid[i + 1]:
        i += 1
    j = 0
    while j < ny - 2 and y > y_grid[j + 1]:
        j += 1
    x1, x2 = x_grid[i], x_grid[i + 1]
    y1, y2 = y_grid[j], y_grid[j + 1]
    Q11 = data[i, j]
    Q21 = data[i + 1, j]
    Q12 = data[i, j + 1]
    Q22 = data[i + 1, j + 1]
    if x2 == x1 or y2 == y1:
        return 0.0
    fxy1 = ((x2 - x) / (x2 - x1)) * Q11 + ((x - x1) / (x2 - x1)) * Q21
    fxy2 = ((x2 - x) / (x2 - x1)) * Q12 + ((x - x1) / (x2 - x1)) * Q22
    fxy = ((y2 - y) / (y2 - y1)) * fxy1 + ((y - y1) / (y2 - y1)) * fxy2
    return fxy

@cuda.jit
def fk_to_fv_kernel(freqs, wavenums, fk_spectrum, freqs_target, vels_target, fv_spectrum):
    j, i = cuda.grid(2)
    nv = vels_target.shape[0]
    nf = freqs_target.shape[0]
    if i < nf and j < nv:
        f = freqs_target[i]
        v = vels_target[j]
        k = f / v if v != 0 else 0.0
        val = bilinear_interp(f, k, freqs, wavenums, fk_spectrum)
        fv_spectrum[j, i] = val


def cal_fk(params, ccdata_t, callback_func, callback_plotdata, callback_str1):
    """
    ccdata: np.ndarray, shape = (Nshots, Nstations, Nt)
    """
    callback_str1("Starting FK->FV calculation...")
    ccdata = _ensure_cc_cube(ccdata_t)[:, 1:, :]
    start_sta = int(np.ceil(params['range'][0]/params['step']))
    end_sta = int(np.floor(params['range'][1]/params['step']))
    fmin = params['fmin']
    fmax = params['fmax']
    vmin = params['vmin']
    vmax = params['vmax']
    dv = params['dv']
    df = params['df']
    dt = params['dt']
    dr = params.get('dr', 1.0)  # 站间距(m),你可以在params里给定
    len_t = int((params['cc_len'] / dt) / 2)
    Nshots, Nstations, Nt = ccdata.shape
    Fs = 1 / params['dt']

    freqs_full = rfftfreq(Nt, d=1/Fs)   # 时间轴频率（正频）
    wavenums_full = fftfreq(Nstations, d=dr)  # 波数，双边

    # 只取正波数部分（因为面波正向）
    pos_wavenum = wavenums_full >= 0
    wavenums = wavenums_full[pos_wavenum].astype(np.float32)

    # 目标频率和速度轴
    freqs_target = np.arange(fmin, fmax + df/2, df).astype(np.float32)
    vels_target = np.arange(vmin, vmax + dv/2, dv).astype(np.float32)
    lv = len(vels_target)
    lf = len(freqs_target)

    fv_all = []

    threadperblock = (16, 16)
    blockspergrid_x = int(np.ceil(lf / threadperblock[0]))
    blockspergrid_y = int(np.ceil(lv / threadperblock[1]))
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    for i in range(0, end_sta - start_sta):
        callback_str1('Calculating disp')
        callback_func(i+start_sta)
        # seisdata_full = np.zeros((ccdata.shape[1], ccdata.shape[-1]))
        # for i_sta in range(ccdata.shape[1]):
        #     seisdata_full[i_sta, :] = ccdata_t[shot_idx, i_sta, :]
        #     seisdata_full[i_sta] /= seisdata_full[i_sta].max()
        # # # seisdata_full[i_sta, int(time_len * sampling) - 4:int(time_len * sampling) + 4] = 0
        #     if len(params['cutpoint']) > 1:
        #         seisdata_full[i_sta,
        #         min(int((len_t + (params['cutpoint'][1][i_sta]) / dt)), seisdata_full.shape[-1]):] = 0
        #         seisdata_full[i_sta,
        #         :max(int((len_t + (params['cutpoint'][0][i_sta]) / dt)), 0)] = 0
        # seisdata_full[:,
        # int(len_t) - 10:int(len_t) + 10] = 0
        seisdata_full = np.zeros((ccdata.shape[1], ccdata.shape[-1]))
        for i_sta in range(ccdata.shape[1]):
            seisdata_full[i_sta, :] = ccdata_t[i, i_sta, :]
            seisdata_full[i_sta] /= seisdata_full[i_sta].max()
        seisdata_full_t = seisdata_full[:,
                        int(len_t) - int(params['time_range'] / params['dt']):int(len_t) + int(
                            params['time_range'] / params['dt'])]
        # if params['Part'] == 'causal':
        #     seisdata = seisdata_full_t[:, int(params['time_range'] / params['dt']):]
        # if params['Part'] == 'noncausal':
        #     seisdata = seisdata_full_t[:, :int(params['time_range'] / params['dt'])]
        #     seisdata = seisdata[:, ::-1]
        # seisdata = np.ascontiguousarray(seisdata)
        # seisdata_fj = np.zeros_like(seisdata)

        # 对空间（Nstations）和时间 (Nt) 做2D FFT（时间用rfft）
        fk_full = np.abs(np.fft.rfft2(seisdata_full_t))

        # 取对应正波数部分
        fk = fk_full[pos_wavenum, :].astype(np.float32)  # shape (wavenums, freqs)

        # 频率轴对应 rfft 频率
        freqs = freqs_full[:fk.shape[1]].astype(np.float32)

        # 转置为 (freqs, wavenums)
        fk_T = fk.T

        # 设备内存
        d_freqs = cuda.to_device(freqs)
        d_wavenums = cuda.to_device(wavenums)
        d_fk_spectrum = cuda.to_device(fk_T)
        d_freqs_target = cuda.to_device(freqs_target)
        d_vels_target = cuda.to_device(vels_target)
        d_fv_spectrum = cuda.to_device(np.zeros((lv, lf), dtype=np.float32))

        # 调用核函数
        fk_to_fv_kernel[blockspergrid, threadperblock](
            d_freqs, d_wavenums, d_fk_spectrum,
            d_freqs_target, d_vels_target, d_fv_spectrum)

        fv_spectrum = d_fv_spectrum.copy_to_host()

        # 归一化每列
        for i in range(fv_spectrum.shape[1]):
            col = fv_spectrum[:, i]
            rng = col.max() - col.min()
            if rng > 0:
                fv_spectrum[:, i] = (col - col.min()) / rng

        # 反转速度轴方便显示
        fv_all.append(fv_spectrum)

    extent = [freqs_target[0], freqs_target[-1], vmin, vmax]
    # plt.figure(figsize=(10, 5))
    # plt.imshow(fv_all[0], extent=extent, aspect='auto', origin='lower', cmap='jet')
    # plt.colorbar(label='Normalized Energy')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Phase Velocity (m/s)')
    # plt.title(f'Shot {i} Frequency-Velocity Dispersion')
    # plt.tight_layout()
    # plt.show()
    callback_plotdata([np.array(fv_all), extent])
    callback_str1("FK->FV calculation finished.")
    return np.array(fv_all), extent
