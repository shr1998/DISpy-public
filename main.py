import os.path
import warnings
import sys
import traceback
import hashlib
import re
import matplotlib
os.environ.setdefault("QT_API", "pyqt5")
matplotlib.use("Qt5Agg")

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# from disba import PhaseSensitivity
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter
import _surf96_vector_gpu
import gsurf96

from ls_inv import _warmup_surf96_safe, invert_surf96_adaptive_mc
from obspy import UTCDateTime
from ruamel.yaml import YAML
import ast

yaml = YAML()
from PyQt5.QtGui import QPixmap, QColor, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QFileDialog, QMessageBox, QGraphicsScene, \
    QWidget, QVBoxLayout, QLabel, \
    QSizePolicy, QPushButton, QTextEdit, QHBoxLayout, QComboBox, QCheckBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QRunnable, QThreadPool
import io
import disp_cal
from torch.utils.data import DataLoader
from SDI import *
from utillis_new_gpu import *
from main_window import Ui_MainWindow
from pick_window import Ui_MainWindow as pick_w
from inv_window import Ui_MainWindow as inv_w
from unet import UNet
from init_model import *
import tqdm
from types import SimpleNamespace
from dispersion_centerline import (
    combine_centerlines,
    extract_dispersion_centerlines,
    order_centerlines_by_min_frequency,
)
from dispersion_params import CC_DATA_PARAMETER_KEYS, resolve_dispersion_params

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "example.yaml")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "unet1.pth")


def _snapshot_path(filename):
    output_dir = os.path.join(PROJECT_ROOT, "output", "screenshots")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

layer_definitions = [
    {
        'd': (0.2, 0.2),
        'vs': (400, 800),
        'density': (1.8, 2.2)
    },
    {
        'd': (0.2, 0.2),
        'vs': (500, 800),
        'density': (2.3, 2.8)
    },
    {
        'd': (0.2, 0.2),
        'vs': (500, 800),
        'density': (2.3, 2.8)
    },
    {
        'd': (0.2, 0.2),
        'vs': (600, 900),
        'density': (2.3, 2.8)
    },
    {
        'd': (0.2, 0.2),
        'vs': (600, 900),
        'density': (2.3, 2.8)
    },
    {
        'd': (0.2, 0.2),
        'vs': (700, 1000),
        'density': (1.8, 2.2)
    },
    {
        'd': (0.2, 0.2),
        'vs': (700, 1000),
        'density': (2.3, 2.8)
    },
    {
        'd': (0.2, 0.2),
        'vs': (800, 1200),
        'density': (2.3, 2.8)
    },
    {
        'd': (0.2, 0.2),
        'vs': (800, 1200),
        'density': (2.3, 2.8)
    },
    {
        'd': (0.2, 0.2),
        'vs': (800, 1200),
        'density': (2.3, 2.8)
    },
]


def has_len(obj):
    try:
        len(obj)
        return True
    except TypeError:
        return False


def sanitize_for_yaml(obj):
    """
    递归转换数据结构中的特殊类型，方便PyYAML保存：
    - UTCDateTime 转成 ISO字符串
    - numpy / cupy 浮点数转成 Python float
    - 递归处理 dict、list、tuple
    """
    if isinstance(obj, dict):
        return {sanitize_for_yaml(k): sanitize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_yaml(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_yaml(i) for i in obj)
    elif isinstance(obj, (np.floating, cp.generic)):
        return float(obj)
    elif isinstance(obj, UTCDateTime):
        return obj.isoformat()
    else:
        return obj


def convert_floats_in_dict(d):
    """
    递归遍历字典，将所有 numpy.float64 / cupy.float64 转为 Python float
    """
    for k, v in d.items():
        if isinstance(v, dict):
            convert_floats_in_dict(v)
        elif isinstance(v, (np.floating, cp.generic)):
            d[k] = float(v)
        # 如果还有其他容器类型需要处理，可以继续扩展
    return d


def update_existing(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1:
            dict1[k] = v
    return dict1


def convert_special_types(obj):
    """递归转换特殊类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: convert_special_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_special_types(item) for item in obj]
    elif isinstance(obj, (ScalarFloat, ScalarInt)):
        return float(obj) if isinstance(obj, ScalarFloat) else int(obj)
    return obj


def extract_numeric_keys(input_dict):
    """提取字典中所有可转换为数值的键
    Args:
        input_dict (dict): 原始字典
    Returns:
        np.ndarray: 数值化的键数组（自动排序）
    """
    numeric_keys = []
    for key in input_dict.keys():
        try:
            # 处理科学计数法 (如 "1e3")
            if isinstance(key, str) and 'e' in key.lower():
                num = float(key)
            # 处理百分数 (如 "85%")
            elif isinstance(key, str) and key.endswith('%'):
                num = float(key[:-1]) / 100
            # 常规转换
            else:
                num = float(key)
            numeric_keys.append(num)
        except (TypeError, ValueError):
            continue  # 静默跳过无效键
    return np.sort(np.array(numeric_keys))


def is_all_integers_np(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    # 排除 NaN、Inf 等非有限值
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        return False
    # 检查所有值是否为整数
    return np.allclose(arr, arr.astype(int))


def dict_to_namespace(data):
    """
    遞歸將字典和列表中的字典轉換為命名空間對象

    參數:
        data: 輸入數據，可以是字典、列表或其他類型

    返回:
        SimpleNamespace對象、處理後的列表或原始數據類型
    """
    if isinstance(data, dict):
        # 處理字典類型，遞歸轉換所有鍵值對
        processed = {k: dict_to_namespace(v) for k, v in data.items()}
        return SimpleNamespace(**processed)
    elif isinstance(data, list):
        # 處理列表類型，遞歸處理每個元素
        return [dict_to_namespace(item) for item in data]
    else:
        # 非容器類型直接返回
        return data


INV_PARAM_KEY_ORDER = [
    "lr",
    "optimizer",
    "es",
    "step_size",
    "gamma",
    "iter_max",
    "inv_object",
    "max_mode",
    "vsrange",
    "inittal_method",
    "init_model_depth",
    "model_num",
    "retry_bad_channel",
    "retry_nstart_factor",
    "retry_vsrange_expand",
    "retry_thickness_expand",
    "disp_resample_mode",
    "disp_resample_points",
    "inv_profile_smooth",
    "ls",
    "device",
    "fmin",
    "fmax",
    "vmax",
    "vmin",
    "step",
    "ML",
]


def _prune_inv_params(params_like):
    params = dict(params_like or {})
    pruned = {}
    for key in INV_PARAM_KEY_ORDER:
        if key in params:
            pruned[key] = params[key]
    pruned["inv_object"] = _resolve_inv_object(params, default=pruned.get("inv_object", "vs"))
    return pruned


def _get_param_value(params_like, key, default=None):
    if params_like is None:
        return default
    if isinstance(params_like, dict):
        return params_like.get(key, default)
    try:
        mapping = vars(params_like)
    except Exception:
        mapping = None
    if isinstance(mapping, dict) and key in mapping:
        return mapping.get(key, default)
    return getattr(params_like, key, default)


def _normalize_inv_object(raw_value):
    raw_text = "" if raw_value is None else str(raw_value).strip()
    if raw_text == "":
        return "vs"
    lowered = raw_text.lower()
    compact = (
        lowered
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("+", "")
        .replace("/", "")
        .replace("&", "and")
    )

    if "厚度" in raw_text and "vs" in lowered:
        return "vsandd"
    if "拟合" in raw_text or "fit" in compact:
        return "vs_fit"
    if compact in {"vs", "vsonly", "shear", "velocitys"}:
        return "vs"
    if compact in {"vsfit", "fit", "bestfit"}:
        return "vs_fit"
    if compact in {"vsandd", "vsd", "joint", "thickness", "thk", "d", "vsandthickness", "vsandthk"}:
        return "vsandd"
    if "vs" in compact and ("thickness" in compact or "andd" in compact or compact.endswith("d")):
        return "vsandd"
    return "vs"


def _resolve_inv_object(params_like, default="vs"):
    preferred_keys = [
        "inversion_target",
        "inv_target",
        "invtarget",
        "target_inv",
        "反演目标",
        "反演对象",
        "inv_object",
        "invobject",
    ]
    for key in preferred_keys:
        value = _get_param_value(params_like, key, None)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return _normalize_inv_object(value)
    return _normalize_inv_object(default)


def _should_invert_thickness(params_like):
    return _resolve_inv_object(params_like, default="vs") == "vsandd"


def get_free_gpu_memory():
    """获取当前 GPU 的剩余显存"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 已分配的显存（MB）
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # 总显存（MB）
        free = total - allocated  # 剩余显存（MB）
        return free
    return 0


def fill_slope_matrix_precise(N, K):
    matrix = [[0 for _ in range(N)] for _ in range(N)]
    # 处理水平线(K=0)
    if K == 0:
        row = N // 2
        for j in range(N):
            matrix[row][j] = 1
        return matrix
    # 处理垂直线(斜率无限大)
    if K == float('inf'):
        col = N // 2
        for i in range(N):
            matrix[i][col] = 1
        return matrix
    # 处理其他斜率
    for i in range(N):
        # 计算对应的j值，四舍五入
        # 直线方程: j - center = K*(i - center)
        center = (N - 1) / 2
        j = round(K * (i - center) + center)
        if 0 <= j < N:
            matrix[i][j] = 1
    return matrix


def custom_median_filter(input_matrix, kernel):
    """
    使用自定义核矩阵进行中值滤波
    参数:
        input_matrix: 输入矩阵 (2D numpy数组)
        kernel: 自定义核矩阵 (2D numpy数组，由0和1组成)
                1表示该位置的像素应被包含在中值计算中
                0表示忽略该位置
    返回:
        滤波后的矩阵
    """
    # 检查核矩阵是否为0和1组成
    if not np.all(np.isin(kernel, [0, 1])):
        raise ValueError("The kernel matrix must contain only 0 and 1")
    # 获取输入矩阵和核的尺寸
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape
    # 计算需要填充的边界大小
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    # 对输入矩阵进行镜像填充
    padded_matrix = np.pad(input_matrix, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    # 初始化输出矩阵
    output = np.zeros_like(input_matrix)
    # 应用中值滤波
    for i in range(input_h):
        for j in range(input_w):
            # 获取当前窗口
            window = padded_matrix[i:i + kernel_h, j:j + kernel_w]
            # 根据核矩阵收集需要考虑的像素值
            values = window[kernel == 1]
            # 计算中值并赋给输出矩阵
            output[i, j] = np.median(values)
    return output


class InteractiveprocessPlot(QWidget):
    callback = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.disp_temp = []
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        # layout.addWidget(self.toolbar)
        self.setLayout(layout)

        # # 添加到工具栏右侧
        # self.toolbar.addSeparator()
        # self.toolbar.addWidget(self.toggle_interaction_btn)
        # # 存储当前显示的散点数据
        # self.current_scatter_data = None
        # self.current_pic_num = None
        # self.current_params = None
        # self.current_extent = None
        # self.current_E_all = None

    def plot(self, c_temp1, t_temp1, A, plot_loc, channelname, c_all=None, t_list=None, draw_mode_lines=True):
        """更新绘图内容"""
        # # 清除前保留坐标轴范围
        # prev_xlim = self.figure.axes[0].get_xlim() if self.figure.axes else None
        # prev_ylim = self.figure.axes[0].get_ylim() if self.figure.axes else None

        self.figure.clear()
        # c = ax.imshow(self.E_all[pic_num], extent=self.extent, aspect='auto', origin='lower', cmap='jet')

        # print('test')

        ax = self.figure.add_subplot(111)
        t_temp1 = 1 / t_temp1
        c_temp1 = np.asarray(c_temp1) * 1000
        if t_list is None:
            t_list = t_temp1
        else:
            t_list = 1 / t_list
        if c_all is None:
            c_all = c_temp1
        else:
            c_all = np.asarray(c_all) * 1000
        c_all = np.asarray(c_all)
        t_list = np.asarray(t_list)
        mode_colors = ["black", "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#8c564b"]
        mode_markers = ["o", "^", "s", "D", "v", "P"]
        AT = has_len(A)
        if not AT:
            for i in range(len(c_temp1)):
                ax.plot(
                    t_temp1,
                    c_temp1[i],
                    color=mode_colors[i % len(mode_colors)],
                    linewidth=1.2,
                    label=f"Mode {i}" if len(c_temp1) > 1 else None,
                )
        else:
            A1 = np.array(A)
            A1[A1 < 0] = -1
            A1[A1 > 0] = 1
            cmap = ListedColormap(["#2cb1a1", "#f28e2b"])  # 0 白色，1 蓝色
            if draw_mode_lines:
                if c_all.ndim == 1:
                    ax.plot(t_list, c_all, color=mode_colors[0], linewidth=1.2, label="Mode 0")
                else:
                    for i in range(len(c_all)):
                        ax.plot(
                            t_list,
                            c_all[i],
                            color=mode_colors[i % len(mode_colors)],
                            linewidth=1.2,
                            label=f"Mode {i}",
                        )
            ax.pcolormesh(t_temp1, c_temp1, A1.T, cmap=cmap, vmin=-1, vmax=1.2)
            # ax.imshow(np.flipud(A1.T))
        plot_loc_arr = np.asarray(plot_loc, dtype=float)
        plotted_mode_points = False
        if plot_loc_arr.ndim == 2 and plot_loc_arr.shape[0] == 2 and plot_loc_arr.shape[1] > 0:
            max_modes_hint = int(len(c_all)) if c_all.ndim > 1 else None
            mode_targets = split_dispersion_modes_2xn(plot_loc_arr, max_modes=max_modes_hint)
            if len(mode_targets) > 1:
                for mode_id, mode_pts in mode_targets.items():
                    ax.scatter(
                        mode_pts[1],
                        mode_pts[0],
                        s=14,
                        color=mode_colors[int(mode_id) % len(mode_colors)],
                        marker=mode_markers[int(mode_id) % len(mode_markers)],
                        edgecolors='white',
                        linewidths=0.3,
                        zorder=3,
                        label=f"Obs M{int(mode_id)}",
                    )
                plotted_mode_points = True
        if not plotted_mode_points:
            ax.scatter(plot_loc[1], plot_loc[0], s=10, color='red', label="Obs")
        # 设置坐标轴标签和标题
        ax.set_xlabel('Frequency[Hz]', fontsize=8)
        ax.set_ylabel('Phase Vel.[m/s]', fontsize=8)
        ax.set_title(f'Channel {int(channelname)} Disp inv')
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            unique = {}
            for h, lab in zip(handles, labels):
                if lab and lab not in unique:
                    unique[lab] = h
            if len(unique) > 1:
                ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
        # self.figure.tight_layout()
        # self.canvas.draw()

        # # 恢复之前的视图范围
        # if prev_xlim and prev_ylim:
        #     ax.set_xlim(prev_xlim)
        #     ax.set_ylim(prev_ylim)

        self.figure.tight_layout()
        self.canvas.draw()


class InteractiveModelPlot(QWidget):
    callback = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(3, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.disp_temp = []
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        # layout.addWidget(self.toolbar)
        self.setLayout(layout)

        # # 添加到工具栏右侧
        # self.toolbar.addSeparator()
        # self.toolbar.addWidget(self.toggle_interaction_btn)
        # # 存储当前显示的散点数据
        # self.current_scatter_data = None
        # self.current_pic_num = None
        # self.current_params = None
        # self.current_extent = None
        # self.current_E_all = None

    def plot(self, x_loc, data2, chanelname):
        """更新绘图内容"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # 绘制图形（这里假设 self.data1 为要绘制的数据）
        ax.step(data2.T, x_loc)
        # 设置坐标轴标签和标题
        ax.invert_yaxis()
        # 配置坐标轴和标题
        ax.set_xlabel('Vel.[km/s]', fontsize=8)
        ax.set_ylabel('Depth[km]', fontsize=8)
        ax.set_title(f'Channel {chanelname} inv result')

        self.figure.tight_layout()
        self.canvas.draw()


class InteractiveFreePlot(QWidget):
    callback = pyqtSignal(np.ndarray)
    callback_cut = pyqtSignal(np.ndarray)
    callback_cut_processed = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4.38, 3.99))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.disp_temp = []
        self.scatter = None
        self.cut_scatter = None
        self._beamforming_cache = None
        self.P_base = None
        layout = QVBoxLayout()
        mode_bar = QHBoxLayout()
        mode_bar.setContentsMargins(0, 0, 0, 0)
        mode_bar.addWidget(QLabel("View"))
        self.analysis_combo = QComboBox(self)
        self.analysis_combo.addItem("FK", "fk")
        self.analysis_combo.addItem("Sensitivity", "sensitivity")
        self.analysis_combo.addItem("Beamforming", "beamforming")
        mode_bar.addWidget(self.analysis_combo)
        mode_bar.addStretch(1)
        layout.addLayout(mode_bar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        # 创建可切换的按钮（代替复选框）
        self.toggle_interaction_btn = QPushButton("PICK")
        self.toggle_interaction_btn.setCheckable(True)  # 使按钮可切换
        self.toggle_interaction_btn.setChecked(True)  # 默认启用
        self.toggle_interaction_btn.toggled.connect(self._update_button_style)
        self.cut_mode_btn = QPushButton("CUT")
        self.cut_mode_btn.setCheckable(False)
        self.cut_mode_btn.clicked.connect(self._on_cut_clicked)

        # 设置按钮样式（亮/暗模式）
        self._update_button_style(self.toggle_interaction_btn.isChecked())
        self._update_cut_mode_style(False)

        # 添加到工具栏右侧
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.toggle_interaction_btn)
        self.toolbar.addWidget(self.cut_mode_btn)
        # 存储当前显示的散点数据
        self.current_scatter_data = None
        self.current_cut_data = None
        self.current_pic_num = None
        self.current_params = None
        self.current_extent = None
        self.current_E_all = None

        # 连接鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_click)
        # self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def get_analysis_mode(self):
        data = self.analysis_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip().lower()
        text = self.analysis_combo.currentText().strip().lower()
        if "beam" in text:
            return "beamforming"
        if "sen" in text:
            return "sensitivity"
        return "fk"

    @staticmethod
    def _get_param_number(params_like, key, default):
        if isinstance(params_like, dict):
            value = params_like.get(key, default)
        else:
            value = getattr(params_like, key, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    def _beamforming_velocity_range_mps(self, params_like=None, disp_params_like=None):
        vmin = self._get_param_number(params_like, "vmin", 100.0)
        vmax = self._get_param_number(params_like, "vmax", 1500.0)
        if isinstance(disp_params_like, dict):
            if "vmin" in disp_params_like:
                vmin = self._get_param_number(disp_params_like, "vmin", vmin)
            if "vmax" in disp_params_like:
                vmax = self._get_param_number(disp_params_like, "vmax", vmax)
        elif disp_params_like is not None:
            if hasattr(disp_params_like, "vmin"):
                vmin = self._get_param_number(disp_params_like, "vmin", vmin)
            if hasattr(disp_params_like, "vmax"):
                vmax = self._get_param_number(disp_params_like, "vmax", vmax)

        # If the range looks like km/s, convert to m/s.
        if vmax <= 20:
            vmin *= 1000.0
            vmax *= 1000.0
        if vmin <= 0:
            vmin = 50.0
        if vmax <= vmin:
            vmax = max(vmin * 1.5, vmin + 100.0)
        return float(vmin), float(vmax)

    def compute_beamforming_spectrum(
            self,
            data_filter,
            dt,
            dx,
            params_like=None,
            disp_params_like=None,
            theta_step=1.0,
            nvel=36,
    ):
        d = np.asarray(data_filter, dtype=float)
        if d.ndim != 2:
            raise ValueError("Beamforming input data must be 2D (receiver x time).")

        nrec, nt = d.shape
        if nrec < 2 or nt < 8:
            raise ValueError("Beamforming requires at least 2 receivers and enough time samples.")

        # Remove per-trace mean to reduce DC leakage.
        d = d - np.mean(d, axis=1, keepdims=True)

        # Frequency axis in NumPy for masking/parameter logic.
        freqs = np.fft.rfftfreq(nt, d=dt)

        fmin = self._get_param_number(params_like, "fmin", 0.0)
        fmax = self._get_param_number(params_like, "fmax", np.max(freqs))
        if isinstance(disp_params_like, dict):
            if "fmin" in disp_params_like:
                fmin = self._get_param_number(disp_params_like, "fmin", fmin)
            if "fmax" in disp_params_like:
                fmax = self._get_param_number(disp_params_like, "fmax", fmax)

        if fmax <= 0:
            fmax = np.max(freqs)
        if fmax <= fmin:
            fmin = 0.0
            fmax = np.max(freqs)

        # Backend: use CuPy when available, otherwise NumPy.
        xp = np
        use_gpu = False
        if isinstance(params_like, dict):
            use_gpu = bool(params_like.get("beamforming_use_gpu", False))
        else:
            use_gpu = bool(getattr(params_like, "beamforming_use_gpu", False))
        if use_gpu:
            try:
                import cupy as _cp
                if _cp.cuda.runtime.getDeviceCount() > 0:
                    xp = _cp
            except Exception:
                xp = np

        # FFT on selected backend.
        d_xp = xp.asarray(d)
        spec = xp.fft.rfft(d_xp, axis=1)  # shape: (nrec, nf)

        # Keep positive frequencies in range and avoid near-DC.
        mask = (freqs >= max(fmin, 0.2)) & (freqs <= fmax)
        f_sel = freqs[mask]
        if f_sel.size == 0:
            # Fallback to an automatic middle band.
            lo = max(0.2, np.percentile(freqs[1:], 10))
            hi = np.percentile(freqs[1:], 60)
            mask = (freqs >= lo) & (freqs <= hi)
            f_sel = freqs[mask]
            if f_sel.size == 0:
                raise ValueError("No valid frequency bins for beamforming.")

        x = xp.arange(nrec, dtype=xp.float64) * float(dx)
        vmin, vmax = self._beamforming_velocity_range_mps(params_like, disp_params_like)

        # A linear array resolves only signed slowness along the array axis. A joint
        # azimuth/velocity scan is non-unique because cos(theta) / velocity is the
        # only observable quantity. Scan that quantity directly instead.
        samples_per_side = int(max(24, nvel * 2))
        p_limit = 1.0 / vmin
        slowness_np = np.linspace(-p_limit, p_limit, 2 * samples_per_side + 1)
        slowness_grid = xp.asarray(slowness_np, dtype=xp.float64)

        # Receiver-normalized snapshots measure spatial coherence at each frequency.
        X_sel = spec[:, mask]  # (nrec, nf_sel)
        denom = xp.linalg.norm(X_sel, axis=0)
        denom = xp.where(denom > 1e-12, denom, 1.0)
        X_sel = X_sel / denom

        power_grid = xp.zeros((f_sel.size, slowness_grid.size), dtype=xp.float64)
        frequency_chunk = 64
        for start in range(0, f_sel.size, frequency_chunk):
            stop = min(start + frequency_chunk, f_sel.size)
            freq_block = xp.asarray(f_sel[start:stop], dtype=xp.float64)
            phase = (
                2.0 * np.pi
                * freq_block[:, None, None]
                * slowness_grid[None, :, None]
                * x[None, None, :]
            )
            steering_conj = xp.exp(1j * phase)
            spectra_block = X_sel[:, start:stop].T[:, None, :]
            beam = xp.sum(steering_conj * spectra_block, axis=2)
            power_grid[start:stop] = (
                beam.real * beam.real + beam.imag * beam.imag
            ) / nrec

        power_curve = xp.mean(power_grid, axis=0)
        peak_idx = int(xp.argmax(power_curve))
        peak_slowness = float(slowness_np[peak_idx])
        if abs(peak_slowness) > 1.0e-12:
            peak_velocity = float(1.0 / peak_slowness)
        else:
            peak_velocity = float("inf")

        max_p = float(xp.max(power_grid))
        if max_p > 0:
            power_grid = power_grid / max_p
            power_curve = power_curve / max_p

        if xp is np:
            power_map_np = np.asarray(power_grid, dtype=float)
            power_curve_np = np.asarray(power_curve, dtype=float)
        else:
            power_map_np = xp.asnumpy(power_grid)
            power_curve_np = xp.asnumpy(power_curve)

        return {
            "slowness_s_per_km": slowness_np * 1000.0,
            "power": power_curve_np,
            "power_map": power_map_np,
            "frequency": f_sel,
            "peak_slowness_s_per_km": peak_slowness * 1000.0,
            "peak_velocity": peak_velocity,
            "f_used": f_sel,
            "v_range_mps": (vmin, vmax),
            "backend": "cupy" if xp is not np else "numpy",
        }

    def plot_beamforming(self, data_filter, dt, dx, pic_num, params_like=None, disp_params_like=None):
        result = self.compute_beamforming_spectrum(
            data_filter=data_filter,
            dt=dt,
            dx=dx,
            params_like=params_like,
            disp_params_like=disp_params_like,
        )
        self._beamforming_cache = result

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="polar")
        slowness = np.asarray(result["slowness_s_per_km"], dtype=float) / 1000.0
        power_curve = np.asarray(result["power"], dtype=float)
        vmin, vmax = result["v_range_mps"]
        theta_deg = np.linspace(0.0, 360.0, 181)
        theta_rad = np.deg2rad(theta_deg)
        velocity = np.linspace(vmin, vmax, 72)

        # Map the observable projected slowness p=cos(theta)/v onto a polar grid.
        projected_slowness = np.cos(theta_rad)[:, None] / velocity[None, :]
        polar_power = np.interp(
            projected_slowness.ravel(),
            slowness,
            power_curve,
        ).reshape(projected_slowness.shape)
        polar_power /= max(float(np.nanmax(polar_power)), 1.0e-12)
        power_db = 10.0 * np.log10(np.maximum(polar_power, 1.0e-2))
        c = ax.contourf(
            theta_rad,
            velocity,
            power_db.T,
            levels=np.linspace(-20.0, 0.0, 41),
            cmap="viridis",
            extend="min",
        )

        peak_slowness = float(result["peak_slowness_s_per_km"])
        peak_v = float(result["peak_velocity"])
        peak_theta = 0.0 if peak_slowness >= 0 else 180.0
        if np.isfinite(peak_v) and vmin <= abs(peak_v) <= vmax:
            ax.scatter(
                np.deg2rad(peak_theta),
                abs(peak_v),
                s=45,
                c="red",
                marker="*",
                zorder=4,
            )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(vmin, vmax)
        ax.set_ylabel("Velocity (m/s)", labelpad=28)
        ax.set_rlabel_position(135)
        ax.grid(True, alpha=0.35)

        if isinstance(params_like, dict):
            step = int(params_like.get("step", 1))
        else:
            step = int(getattr(params_like, "step", 1))
        ax.set_title(
            f'Ch {int(pic_num * step)} | '
            f'Az={peak_theta:.0f}° | Vapp={abs(peak_v):.0f} m/s',
            fontsize=10,
        )
        cbar = self.figure.colorbar(c, ax=ax, pad=0.02)
        cbar.set_label("Relative Beam Power (dB)")
        self.figure.tight_layout()
        self.canvas.draw()

    def _update_button_style(self, checked):
        """根据按钮状态更新样式（亮/暗）"""
        if checked:
            # "亮"模式（启用交互）
            self.toggle_interaction_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;  /* 绿色 */
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.toggle_interaction_btn.setText("PICK ✔")
        else:
            # "暗"模式（禁用交互）
            self.toggle_interaction_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;  /* 红色 */
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            self.toggle_interaction_btn.setText("PICK ✖")

    def _update_cut_mode_style(self, checked):
        _ = checked
        self.cut_mode_btn.setStyleSheet("")
        self.cut_mode_btn.setText("CUT")

    def _on_cut_clicked(self):
        # FK CUT click: no operation.
        return

    def _active_point_mode(self):
        return "pick"

    def _build_fk_cut_mask(self, k_axis, f_axis, cut_points):
        pts = np.asarray(cut_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
            return None

        k_pts = pts[:, 0]
        f_pts = pts[:, 1]
        pos_mask = k_pts > 0
        neg_mask = k_pts < 0
        if np.count_nonzero(pos_mask) < 2 or np.count_nonzero(neg_mask) < 2:
            return None

        try:
            slope_pos, intercept_pos = np.polyfit(f_pts[pos_mask], k_pts[pos_mask], 1)
            slope_neg, intercept_neg = np.polyfit(f_pts[neg_mask], k_pts[neg_mask], 1)
        except Exception:
            return None

        k_pos = f_axis * slope_pos + intercept_pos
        k_neg = f_axis * slope_neg + intercept_neg
        k_pos = np.maximum(k_pos, 0.0)
        k_neg = np.minimum(k_neg, 0.0)

        mask = np.zeros((len(k_axis), len(f_axis)), dtype=float)
        for jf in range(len(f_axis)):
            left = min(k_neg[jf], k_pos[jf])
            right = max(k_neg[jf], k_pos[jf])
            keep = (k_axis >= left) & (k_axis <= right)
            mask[:, jf] = keep.astype(float)
        return mask

    def apply_fk_cut_and_redraw(self):
        if self.get_analysis_mode() != "fk":
            return
        if self.P_base is None or self.current_cut_data is None:
            return
        if not hasattr(self, "lf") or not hasattr(self, "freq_window"):
            return

        f_start = int(self.lf / 2)
        f_stop = int(self.lf / 2 + self.freq_window)
        if f_stop <= f_start:
            return

        k_axis = np.asarray(self.k, dtype=float)
        f_axis = np.asarray(self.f[f_start:f_stop], dtype=float)
        mask = self._build_fk_cut_mask(k_axis, f_axis, self.current_cut_data)
        if mask is None:
            return

        self.P = np.asarray(self.P_base, dtype=float).copy()
        self.P[:, f_start:f_stop] *= mask
        self.callback_cut_processed.emit(np.asarray(self.P, dtype=float))
        if self.current_pic_num is not None and self.current_params is not None:
            self.plot(self.current_pic_num, self.current_params)

    def plot(self, pic_num, params):
        """更新绘图内容"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.pcolormesh(self.k, self.f[int(self.lf / 2):int(self.lf / 2 + self.freq_window)],
                      self.P[:, int(self.lf / 2):int(self.lf / 2 + self.freq_window)].T, cmap='viridis')
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Frequancy(Hz)')
        ax.set_title(f'Channel {int(pic_num * params["step"])} F-K')
        self.figure.tight_layout()
        self.canvas.draw()

    def _phase_velocity_single(self, period, mode, thickness, vp, vs, rho):
        period_arr = np.asarray([float(period)], dtype=float)
        thk = np.maximum(np.asarray(thickness, dtype=float), 1.0e-4)
        disp = gsurf96.surf96(
            period_arr,
            thk,
            np.asarray(vp, dtype=float),
            np.asarray(vs, dtype=float),
            np.asarray(rho, dtype=float),
            itype=0,
            mode=int(mode),
        )
        disp = np.asarray(disp, dtype=float).reshape(-1)
        if disp.size == 0 or not np.isfinite(disp[0]):
            raise ValueError("Invalid dispersion result from surf96")
        return float(disp[0])

    def _finite_difference_kernel(
            self,
            param_name,
            thickness,
            vp,
            vs,
            rho,
            period=20.0,
            mode=0,
            rel_step=0.03,
            normalize=True,
    ):
        thickness = np.asarray(thickness, dtype=float).copy()
        vp = np.asarray(vp, dtype=float).copy()
        vs = np.asarray(vs, dtype=float).copy()
        rho = np.asarray(rho, dtype=float).copy()
        nlayer = len(vs)
        kernel = np.zeros(nlayer, dtype=float)

        for i in range(nlayer):
            thk_p = thickness.copy()
            vp_p = vp.copy()
            vs_p = vs.copy()
            rho_p = rho.copy()
            thk_m = thickness.copy()
            vp_m = vp.copy()
            vs_m = vs.copy()
            rho_m = rho.copy()

            if param_name == "thickness":
                delta = max(abs(thickness[i]) * rel_step, 1.0e-4)
                thk_p[i] = max(thk_p[i] + delta, 1.0e-4)
                thk_m[i] = max(thk_m[i] - delta, 1.0e-4)
                delta_eff = thk_p[i] - thk_m[i]
            elif param_name == "vp":
                delta = max(abs(vp[i]) * rel_step, 1.0e-4)
                vp_p[i] = max(vp_p[i] + delta, 1.0e-4)
                vp_m[i] = max(vp_m[i] - delta, 1.0e-4)
                delta_eff = vp_p[i] - vp_m[i]
            elif param_name == "vs":
                delta = max(abs(vs[i]) * rel_step, 1.0e-4)
                vs_p[i] = max(vs_p[i] + delta, 1.0e-4)
                vs_m[i] = max(vs_m[i] - delta, 1.0e-4)
                delta_eff = vs_p[i] - vs_m[i]
            elif param_name == "rho":
                delta = max(abs(rho[i]) * rel_step, 1.0e-4)
                rho_p[i] = max(rho_p[i] + delta, 1.0e-4)
                rho_m[i] = max(rho_m[i] - delta, 1.0e-4)
                delta_eff = rho_p[i] - rho_m[i]
            else:
                raise ValueError(f"Unsupported parameter: {param_name}")

            try:
                c_plus = self._phase_velocity_single(period, mode, thk_p, vp_p, vs_p, rho_p)
                c_minus = self._phase_velocity_single(period, mode, thk_m, vp_m, vs_m, rho_m)
                if delta_eff <= 0:
                    kernel[i] = np.nan
                else:
                    deriv = (c_plus - c_minus) / delta_eff
                    if normalize:
                        if param_name == "thickness":
                            m0 = max(float(thickness[i]), 1.0e-4)
                        elif param_name == "vp":
                            m0 = max(float(vp[i]), 1.0e-4)
                        elif param_name == "vs":
                            m0 = max(float(vs[i]), 1.0e-4)
                        elif param_name == "rho":
                            m0 = max(float(rho[i]), 1.0e-4)
                        else:
                            m0 = 1.0
                        c0 = max(abs(0.5 * (c_plus + c_minus)), 1.0e-8)
                        kernel[i] = deriv * (m0 / c0)
                    else:
                        kernel[i] = deriv
            except Exception:
                kernel[i] = np.nan

        kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
        return kernel

    def _srfker96_local(
            self,
            param_name,
            thickness,
            vp,
            vs,
            rho,
            period=20.0,
            mode=0,
            rel_step=0.03,
    ):
        """
        Local srfker96-style kernel interface.
        Returns layer-wise normalized kernel dlnc/dlnm.
        """
        if "disba" not in sys.modules:
            import numba

            disable_jit = numba.config.DISABLE_JIT
            numba.config.DISABLE_JIT = 1
            try:
                from disba import PhaseSensitivity
            finally:
                numba.config.DISABLE_JIT = disable_jit
        else:
            from disba import PhaseSensitivity

        p = str(param_name).strip().lower()
        alias_map = {
            "d": "thickness",
            "thk": "thickness",
            "thickness": "thickness",
            "a": "vp",
            "alpha": "vp",
            "vp": "vp",
            "b": "vs",
            "beta": "vs",
            "vs": "vs",
            "rho": "rho",
            "density": "rho",
        }
        p = alias_map.get(p, p)
        disba_parameter = {
            "thickness": "thickness",
            "vp": "velocity_p",
            "vs": "velocity_s",
            "rho": "density",
        }.get(p)
        if disba_parameter is None:
            raise ValueError(f"Unsupported sensitivity parameter: {param_name}")

        thickness = np.asarray(thickness, dtype=float)
        vp = np.asarray(vp, dtype=float)
        vs = np.asarray(vs, dtype=float)
        rho = np.asarray(rho, dtype=float)
        calculator = PhaseSensitivity(
            thickness,
            vp,
            vs,
            rho,
            algorithm="dunkin",
            dp=float(rel_step),
        )
        result = calculator(
            float(period),
            mode=int(mode),
            wave="rayleigh",
            parameter=disba_parameter,
        )
        base_model = {
            "thickness": thickness,
            "vp": vp,
            "vs": vs,
            "rho": rho,
        }[p]
        phase_velocity = max(abs(float(result.velocity)), 1.0e-12)
        return np.asarray(result.kernel, dtype=float) * base_model / phase_velocity

    def _kernel_step_curve(self, kernel, thickness):
        kernel = np.asarray(kernel, dtype=float)
        thickness = np.asarray(thickness, dtype=float)
        depth_edges = np.concatenate(([0.0], np.cumsum(thickness)))
        x_vals = np.repeat(kernel, 2)
        y_vals = np.repeat(depth_edges[:-1], 2)
        y_vals = np.concatenate((y_vals, [depth_edges[-1]]))
        x_vals = np.concatenate(([kernel[0]], x_vals))
        return x_vals, y_vals

    def _select_sensitivity_period(self, velocity_model_thickness, velocity_model_vs):
        total_depth = float(np.sum(np.asarray(velocity_model_thickness, dtype=float)))
        mean_vs = float(np.mean(np.asarray(velocity_model_vs, dtype=float)))
        if mean_vs <= 0:
            return 20.0
        est_period = 2.2 * total_depth / mean_vs
        return float(np.clip(est_period, 0.5, 20.0))

    def _composite_kernel_over_periods(
            self,
            param_name,
            thickness,
            vp,
            vs,
            rho,
            mode,
            periods,
            normalize=True,
    ):
        def _density_weights_from_periods(period_arr):
            period_arr = np.asarray(period_arr, dtype=float).reshape(-1)
            period_arr = period_arr[np.isfinite(period_arr) & (period_arr > 0)]
            if period_arr.size <= 1:
                return period_arr, np.ones_like(period_arr, dtype=float)

            # Use frequency spacing compensation:
            # sparse frequency points get larger weights, dense clusters get smaller weights.
            freq = 1.0 / period_arr
            uniq_f, inv = np.unique(freq, return_inverse=True)
            if uniq_f.size <= 1:
                w_uniq = np.ones_like(uniq_f, dtype=float)
            else:
                order = np.argsort(uniq_f)
                f_sorted = uniq_f[order]
                delta = np.zeros_like(f_sorted, dtype=float)
                delta[0] = max(f_sorted[1] - f_sorted[0], 1.0e-12)
                delta[-1] = max(f_sorted[-1] - f_sorted[-2], 1.0e-12)
                if f_sorted.size > 2:
                    delta[1:-1] = np.maximum(
                        0.5 * (f_sorted[2:] - f_sorted[:-2]),
                        1.0e-12
                    )
                w_sorted = delta
                w_uniq = np.zeros_like(w_sorted, dtype=float)
                w_uniq[order] = w_sorted

            w = w_uniq[inv]
            w = np.maximum(w, 1.0e-12)
            w /= np.sum(w)
            return period_arr, w

        periods = np.asarray(periods, dtype=float).reshape(-1)
        periods = periods[np.isfinite(periods) & (periods > 0)]
        if periods.size == 0:
            return self._finite_difference_kernel(
                param_name,
                thickness,
                vp,
                vs,
                rho,
                period=self._select_sensitivity_period(thickness, vs),
                mode=mode,
                normalize=normalize,
            )
        periods, weights = _density_weights_from_periods(periods)
        kernels = []
        for per in periods:
            k = self._srfker96_local(
                param_name,
                thickness,
                vp,
                vs,
                rho,
                period=float(per),
                mode=mode,
            )
            if not normalize:
                k = np.asarray(k, dtype=float)
            kernels.append(np.asarray(k, dtype=float))
        if len(kernels) == 0:
            return np.zeros_like(vs, dtype=float)
        k_mat = np.vstack(kernels)
        valid = np.isfinite(k_mat)
        w_col = weights.reshape(-1, 1)
        num = np.nansum(np.where(valid, k_mat, 0.0) * w_col, axis=0)
        den = np.sum(np.where(valid, 1.0, 0.0) * w_col, axis=0)
        out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return out

    def plot_sen(self, pic_num, velocity_model, mode=0, param_filter="all", periods=None):
        velocity_model_thickness = np.asarray(velocity_model[0], dtype=float)
        velocity_model_vp = np.asarray(velocity_model[1], dtype=float)
        velocity_model_vs = np.asarray(velocity_model[2], dtype=float)
        velocity_model_rho = np.asarray(velocity_model[3], dtype=float)
        mode = int(mode)
        if periods is None:
            period = self._select_sensitivity_period(velocity_model_thickness, velocity_model_vs)
            periods = np.array([period], dtype=float)
        else:
            periods = np.asarray(periods, dtype=float).reshape(-1)
            periods = periods[np.isfinite(periods) & (periods > 0)]
            if periods.size == 0:
                period = self._select_sensitivity_period(velocity_model_thickness, velocity_model_vs)
                periods = np.array([period], dtype=float)
            else:
                period = float(np.median(periods))

        kernel_args = (
            velocity_model_thickness,
            velocity_model_vp,
            velocity_model_vs,
            velocity_model_rho,
        )
        kernel_thk = self._srfker96_local("thickness", *kernel_args, period=period, mode=mode)
        kernel_vp = self._srfker96_local("vp", *kernel_args, period=period, mode=mode)
        kernel_vs = self._srfker96_local("vs", *kernel_args, period=period, mode=mode)
        kernel_rho = self._srfker96_local("rho", *kernel_args, period=period, mode=mode)
        """更新绘图内容"""
        self.figure.clear()
        ax1 = self.figure.add_subplot(111)
        ##1.Rayleigh-wave sensitivity kernels
        kernel_map = {
            "thickness": (kernel_thk, "blue", r'${\partial \ln c}/{\partial \ln d}$'),
            "vp": (kernel_vp, "orange", r'${\partial \ln c}/{\partial \ln \alpha}$'),
            "vs": (kernel_vs, "green", r'${\partial \ln c}/{\partial \ln \beta}$'),
            "rho": (kernel_rho, "red", r'${\partial \ln c}/{\partial \ln \rho}$'),
        }
        selected_keys = ["thickness", "vp", "vs", "rho"]
        if isinstance(param_filter, str):
            pf = param_filter.strip().lower()
            if pf in kernel_map:
                selected_keys = [pf]
            elif pf not in {"", "all", "全部"}:
                selected_keys = [k for k in pf.replace(" ", "").split(",") if k in kernel_map]
                if len(selected_keys) == 0:
                    selected_keys = ["thickness", "vp", "vs", "rho"]

        all_kernel_vals = np.hstack([kernel_map[k][0] for k in selected_keys])
        finite_vals = np.abs(all_kernel_vals[np.isfinite(all_kernel_vals)])
        if finite_vals.size:
            k_abs = float(np.nanpercentile(finite_vals, 97.0))
        else:
            k_abs = 1.0
        k_abs = max(k_abs, 1.0e-4)
        ax1.set_xlim(-1.1 * k_abs, 1.1 * k_abs)
        ax1.set_xlabel("Normalized sensitivity (dlnc / dlnm)", fontsize=15)
        ax1.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax1.set_ylim(float(np.sum(velocity_model_thickness)), 0.0)
        ax1.set_ylabel("Depth[km]", fontsize=15)
        ax1.grid(axis="y")
        ax1.set_title(
            f"Channel {pic_num}  mode={mode}  period={period:.3f}s  params={','.join(selected_keys)}",
            fontsize=20
        )
        for key in selected_keys:
            k_arr, color, label = kernel_map[key]
            x_k, y_k = self._kernel_step_curve(k_arr, velocity_model_thickness)
            ax1.plot(x_k, y_k, color=color, linewidth=1, label=label)
        ax1.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax1.legend(loc="lower right", fontsize=10)
        self.figure.tight_layout()
        self.canvas.draw()

    def fk_filter(self, data_filter, dt, dx, w=5, show=True, freq_window=5):
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
        self.f = np.arange(-n // 2, n // 2) * fs / (n - 1)
        # 波数 (每米)。
        self.k = 2 * np.pi * np.arange(-m // 2, m // 2) * xs / (m - 1)
        # 二维FFT。
        fk = np.fft.fft2(D)
        if w:
            N = 9
            K = 1  # 45度对角线
            filter_window = fill_slope_matrix_precise(N, K)
            mask = np.zeros(fk.shape)
            h = len(mask)
            l = len(mask[1])
            temp = (w * h / 2) / l
            for i in range(len(mask[0])):
                # mask[max(int(h / 2), int(round(h / 2 - i * temp))):max(int(h / 2 + 1), int(round(h / 2 + i * temp))), i] = 0
                if i > len(mask[0]) / 2:
                    mask[0:max(0, int(round((l - i) * temp))), l - i - 1] = 1
                    mask[max(0, int(round(h - (l - i) * temp))):h, l - i - 1] = 1
                else:
                    mask[0:max(0, int(round(abs((l - i) * temp - temp * l)))), l - i - 1] = 1
                    mask[max(0, int(round(h - abs((l - i) * temp - temp * l)))):h, l - i - 1] = 1
            fk = fk * mask
        if show:
            self.freq_window = freq_window / dt
            fk_1 = fk.copy()
            self.pmin = -10
            self.fs = 1 / dt
            self.lf = len(self.f)
            P = abs(np.fft.fftshift(fk_1))
            # N = 9
            # K = 0.1  # 45度对角线
            # filter_window = fill_slope_matrix_precise(N, K)
            # P = custom_median_filter(P, filter_window)
            self.P = P / P.max()
            self.P_base = self.P.copy()
            # self.P = 10 * np.log10(P)
            P2 = abs(fk)
            P2 /= P2.max()
            # P2 = 10 * np.log10(P2)

        PP = np.fft.ifft2(fk)
        return PP.real

    def on_click(self, event):
        if self.get_analysis_mode() != "fk" and (not self.toggle_interaction_btn.isChecked()):
            return
        """处理鼠标点击事件"""
        if event.inaxes:
            if event.xdata is None or event.ydata is None:
                return
            x, y = round(event.xdata, 2), round(event.ydata, 2)

            # 左键添加点
            if event.button == 1:  # 左键
                self.add_point(x, y)

            # 右键删除点
            elif event.button == 3:  # 右键
                self.delete_nearest_point(x, y)

    # def on_motion(self, event):
    #     """鼠标移动事件，用于实时显示坐标"""
    #     if event.inaxes == self.ax:
    #         x = round(event.xdata, 2)
    #         y = round(event.ydata, 2)
    #         self.coord_label.setText(f"X: {x:.2f}, Y: {y:.2f}")  # 更新工具栏标签
    #     else:
    #         self.coord_label.setText("X: -, Y: -")  # 鼠标移出时清空
    # if event.inaxes:
    #     x, y = round(event.xdata, 2), round(event.ydata, 2)
    #     # 可以在这里添加实时坐标显示功能

    def add_point(self, x, y):

        """在指定坐标添加新点"""
        active_mode = self._active_point_mode()
        if active_mode == "cut":
            if self.current_cut_data is None:
                self.current_cut_data = np.empty((0, 2))
            target = self.current_cut_data
        else:
            if self.current_scatter_data is None:
                self.current_scatter_data = np.empty((0, 2))
            target = self.current_scatter_data

        # 添加新点
        new_point = np.array([[x, y]])
        target = np.concatenate(
            (target, new_point),
            axis=0
        )
        if active_mode == "cut":
            self.current_cut_data = target
        else:
            self.current_scatter_data = target

        # 重新绘制
        self.redraw_plot()
        print(f"Point added: ({x:.2f}, {y:.2f})")

    def delete_nearest_point(self, x, y):
        """删除距离点击位置最近的点（基于像素距离）"""
        active_mode = self._active_point_mode()
        target = self.current_cut_data if active_mode == "cut" else self.current_scatter_data
        if target is None or len(target) == 0:
            return

        # 获取坐标轴和转换器
        ax = self.figure.axes[0]
        trans = ax.transData

        # 将点击位置转换为显示坐标
        click_display = trans.transform((x, y))

        # 计算所有点到点击位置的像素距离
        min_dist = float('inf')
        nearest_idx = -1

        for i, point in enumerate(target):
            # 将数据点转换为显示坐标
            point_display = trans.transform((point[0], point[1]))

            # 计算像素距离
            dist = np.sqrt((click_display[0] - point_display[0]) ** 2 +
                           (click_display[1] - point_display[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx >= 0:
            # 获取要删除的点坐标
            deleted_point = target[nearest_idx]

            # 删除最近的点
            target = np.delete(target, nearest_idx, axis=0)
            if active_mode == "cut":
                self.current_cut_data = target
            else:
                self.current_scatter_data = target

            # 重新绘制
            self.redraw_plot()
            print(f"Point removed: ({deleted_point[0]:.2f}, {deleted_point[1]:.2f})")

    def redraw_plot(self):
        # In FK mode, PICK should only edit points; computation is triggered by CUT button.
        if self.get_analysis_mode() != "fk":
            self.callback.emit(self.current_scatter_data)
        self.callback_cut.emit(self.current_cut_data if self.current_cut_data is not None else np.empty((0, 2)))
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        # 更新散点
        if self.scatter:
            self.scatter.remove()
        if self.cut_scatter:
            self.cut_scatter.remove()

        if len(self.current_scatter_data) > 0:
            self.scatter = ax.scatter(self.current_scatter_data[:, 0],
                                      self.current_scatter_data[:, 1],
                                      s=10,
                                      c='white',
                                      picker=5)
        else:
            self.scatter = None

        if self.current_cut_data is not None and len(self.current_cut_data) > 0:
            self.cut_scatter = ax.scatter(self.current_cut_data[:, 0],
                                          self.current_cut_data[:, 1],
                                          s=14,
                                          c='#00ffff',
                                          marker='x',
                                          picker=5)
        else:
            self.cut_scatter = None

        # 仅更新画布不重建
        self.canvas.draw_idle()


class invprofileWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scatter = None
        self.count = True
        self.figure = Figure(figsize=(20, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)


    def plot(self, x_new, y_new, v_interp):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        x_grid = np.asarray(x_new, dtype=float)
        y_grid = np.asarray(y_new, dtype=float) / 10.0
        z_grid = np.asarray(v_interp, dtype=float).T

        # 伪彩色图
        norm = colors.Normalize(
            vmin=np.nanmin(v_interp),
            vmax=np.nanmax(v_interp)
        )

        mesh = self.ax.pcolormesh(
            x_grid, y_grid, z_grid,
            cmap='jet', shading='auto',
            edgecolors='face',
            antialiased=True,
            norm=norm
        )

        z_min = float(np.nanmin(z_grid))
        z_max = float(np.nanmax(z_grid))
        # if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
        #     contour_levels = np.linspace(z_min, z_max, 8)
        #     contours = self.ax.contour(
        #         x_grid,
        #         y_grid,
        #         z_grid,
        #         levels=contour_levels,
        #         colors='k',
        #         linewidths=0.35,
        #         alpha=0.65,
        #     )
        #     self.ax.clabel(contours, inline=True, fontsize=3, fmt="%.2f")

        # ✅ 正确添加 colorbar（每次都加）
        cbar = self.figure.colorbar(mesh, ax=self.ax)
        cbar.set_label('Vel(km/s)', fontsize=4)

        self.ax.set_title("Inv profile", fontsize=6, pad=10)
        self.ax.set_xlabel("Channel", fontsize=6, labelpad=5)
        self.ax.xaxis.set_label_position('top')
        self.ax.set_ylabel("Depth (km)", fontsize=4, labelpad=5)
        self.ax.invert_yaxis()

        self.ax.tick_params(axis='x', which='both',
                            bottom=False, labelbottom=False,
                            top=True, labeltop=True)
        self.ax.tick_params(axis='x', labelsize=5)
        self.ax.tick_params(axis='y', labelsize=4)

        self.figure.tight_layout()

        # 刷新画布
        self.canvas.draw()


class RawNCFPlotWidget(QWidget):
    callback = pyqtSignal(np.ndarray)
    callback1 = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scatter = None
        self.figure = Figure(figsize=(4.79, 7.92))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.reset_btn = QPushButton("line")
        self.reset_btn.setCheckable(True)  # 使按钮可切换
        self.reset_btn.setChecked(True)  # 默认启用
        self.reset_btn.toggled.connect(self.change_plot)
        self.change_plot(self.reset_btn.isChecked())
        # 设置布局

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # 创建可切换的按钮（代替复选框）
        self.toggle_interaction_btn = QPushButton("CUT")
        self.toggle_interaction_btn.setCheckable(True)  # 使按钮可切换
        self.toggle_interaction_btn.setChecked(True)  # 默认启用
        self.toggle_interaction_btn.toggled.connect(self._update_button_style)

        # 设置按钮样式（亮/暗模式）
        self._update_button_style(self.toggle_interaction_btn.isChecked())
        # 添加到工具栏右侧

        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.toggle_interaction_btn)
        # 添加到工具栏（就在 CUT 按钮旁边）
        self.toolbar.addWidget(self.reset_btn)

        # 可选：绑定事件
        self.reset_btn.clicked.connect(self.change_plot)
        # 初始化属性
        self.ax = None
        self.current_data = None
        self.current_pic_num = None
        self.current_scatter_data = None
        # 连接鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.setStyleSheet("""
        RawNCFPlotWidget{
        border: 1px solid black
        }""")

    def change_plot(self, checked):
        """根据按钮状态更新样式（亮/暗）"""
        if checked:
            self.reset_btn.setText("LINE")
            self.reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;  /* 绿色 */
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
        else:
            # "暗"模式（禁用交互）
            self.reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #E67E22;  /* 红色 */
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            self.reset_btn.setText("IMSHOW")

    def _update_button_style(self, checked):
        """根据按钮状态更新样式（亮/暗）"""
        if checked:
            # "亮"模式（启用交互）
            self.toggle_interaction_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50 ; 
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.toggle_interaction_btn.setText("CUT")
        else:
            # "暗"模式（禁用交互）
            self.toggle_interaction_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;  
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            self.toggle_interaction_btn.setText("NCUT")

    def plot(self, cc_data, pic_num, params):
        """更新绘图内容"""
        self.current_pic_num = pic_num
        self.figure.clear()

        # 创建坐标轴
        self.ax = self.figure.add_subplot(111)

        # 计算坐标轴参数
        dt = 1 / params['samplerate_in_use']
        len_t = int((params['cc_len'] / dt) / 2)
        dr = params.get('dr', 1)  # 默认间距为1

        # 生成坐标轴数据
        r_scale = [tra * dr for tra in range(cc_data.shape[-2])]
        t_scalef = [i * dt for i in range(
            -int(params['time_range'] / dt),
            int(params['time_range'] / dt)
        )]
        cc = cc_data[int(self.current_pic_num)]
        # st = []
        # stats = {'delta': 0.02}
        # for i in range(len(cc)):
        #     tr = Trace(data=cc[i], header=stats)
        #     st.append(tr)
        #     st = Stream(st)
        # st = st.filter('lowpass', freq=2)
        # st = st.filter('bandstop', freqmin=0.1, freqmax=0.25,corners=4)
        # cc = np.array(st)
        seisdata_full = cc.copy()

        cutpoint = np.asarray(params.get('cutpoint', []), dtype=float)
        has_cutpoint = (cutpoint.ndim == 2 and cutpoint.shape[0] == 2 and cutpoint.shape[1] > 0)
        if has_cutpoint:
            ncut = cutpoint.shape[1]
            for i_sta in range(cc.shape[0]):
                # seisdata_full[i_sta] = cc[i_sta, len_t - int(params['time_range'] / dt):len_t + int(
                #     params['time_range'] / dt)]
                amp = float(np.max(np.abs(seisdata_full[i_sta])))
                if amp > 0:
                    seisdata_full[i_sta] /= amp
                # seisdata_full[i_sta, int(time_len * sampling) - 4:int(time_len * sampling) + 4] = 0
                cut_idx = min(i_sta, ncut - 1)
                t_neg = float(cutpoint[0, cut_idx])
                t_pos = float(cutpoint[1, cut_idx])
                if np.isfinite(t_neg) and np.isfinite(t_pos):
                    right_start = min(int(len_t + t_pos / dt), seisdata_full.shape[-1])
                    left_end = max(int(len_t + t_neg / dt), 0)
                    seisdata_full[i_sta, right_start:] = 0
                    seisdata_full[i_sta, :left_end] = 0
                # seisdata  _full[i_sta, int(time_len * sampling) - 48:int(time_len * sampling) + 80] = 0
                ##########################################################################################
                # positive
                # seisdata[i_sta, :] = seisdata_full[i_sta, int(time_len * sampling):]
                # seisdata[i_sta, :32] = 0
                ##########################################################################################
                # negative
                # seisdata[i_sta, :] = st[0].data[int(npts/2-time_len * sampling):int(npts/2)]
                # seisdata[i_sta, :] = seisdata[i_sta, ::-1]
                # seisdata[i_sta, - 8:] = 0
                c0 = max(int(len_t) - 10, 0)
                c1 = min(int(len_t) + 10, seisdata_full.shape[-1])
                seisdata_full[i_sta, c0:c1] = 0

        # for i_sta in range(cc.shape[0]):
        #     seisdata_full[i_sta,
        #     int(params['time_range'] / dt) - round(i_sta) - 250:int(
        #         params['time_range'] / dt) + 250 + round(i_sta)] = 0
        self.callback1.emit(seisdata_full)
        # 提取并标准化数据
        data_slice = seisdata_full[:,
                     len_t - int(params['time_range'] / dt):len_t + int(params['time_range'] / dt)]
        self.current_data = data_slice
        # self.current_data = np.flip(self.current_data, axis=0)
        # # 绘制伪彩色图
        # mesh = self.ax.pcolormesh(t_scalef, r_scale, self.current_data,
        #                               vmax=1, vmin=-1, cmap='seismic')
        if self.reset_btn.isChecked():
            if len(self.current_data) > 100:
                for j in range(0, len(self.current_data), 2):
                    if j % 1 == 0:
                        data = self.current_data[j]
                        # dist = st[0].stats.sac.dist*1000
                        # dist = j * 4
                        dist = r_scale[j]
                        d_max = max(abs(data))
                        data_plot = [i * 30 / d_max + dist for i in data]
                        if j == 2:
                            self.ax.plot(t_scalef, data_plot, 'red', lw=1.2)
                            # continue
                        else:
                            self.ax.plot(t_scalef, data_plot, 'black', lw=1.2)
            else:
                for j in range(0, len(self.current_data)):
                    if j % 1 == 0:
                        data = self.current_data[j]
                        # dist = st[0].stats.sac.dist*1000
                        # dist = j * 4
                        dist = r_scale[j]
                        d_max = max(abs(data)) + 1
                        data_plot = [i * 3 / d_max + dist for i in data]
                        # if j <=5:
                        #     data_plot = [i * 3 / d_max + dist for i in data]
                        # if j > 5:
                        #     data_plot = [i * 90 / d_max + dist for i in data]
                        if j == 2:
                            self.ax.plot(t_scalef, data_plot, 'red', lw=1.2)
                            # continue
                        else:
                            self.ax.plot(t_scalef, data_plot, 'black', lw=1.2)
        else:
            # self.ax.imshow(self.current_data, cmap='seismic', aspect='auto')
            mesh = self.ax.pcolormesh(t_scalef, r_scale[1:], self.current_data[1:],
                                      cmap='seismic')
        self.scatter = None
        # if self.current_scatter_data:
        #     # try:
        #         key = str(pic_num * int(params['step']))
        #         self.scatter = self.ax.scatter(self.current_scatter_data[:, 0],
        #                                   self.current_scatter_data[:, 1],
        #                                   s=10,
        #                                   c='white',
        #                                   picker=5)
        # except KeyError:
        #     self.current_scatter_data = np.empty((0, 2))
        # self.figure.colorbar(mesh, ax=self.ax)
        self.ax.set_xlabel('Time/s', fontsize=12)
        self.ax.set_ylabel('Distance/m', fontsize=12)
        self.ax.set_title(f'Channel {self.current_pic_num * int(params["step"])} RAWNCF')

        # 设置刻度字体
        self.ax.tick_params(axis='both', labelsize=10)

        # 自动调整布局
        self.figure.tight_layout()

        # 刷新画布
        self.canvas.draw()

    def on_click(self, event):
        if not self.toggle_interaction_btn.isChecked():
            return
        """处理鼠标点击事件"""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            print(x)
            # 左键添加点
            if event.button == 1:  # 左键
                self.add_point(x, y)

            # 右键删除点
            elif event.button == 3:  # 右键
                self.delete_nearest_point(x, y)

    def add_point(self, x, y):
        """在指定坐标添加新点"""
        if self.current_scatter_data is None:
            self.current_scatter_data = np.empty((0, 2))

        # 添加新点
        new_point = np.array([[x, y]])
        self.current_scatter_data = np.vstack((self.current_scatter_data, new_point))

        # 重新绘制
        self.redraw_plot()
        print(f"Point added: ({x:.2f}, {y:.2f})")

    def delete_nearest_point(self, x, y):
        """删除距离点击位置最近的点（基于像素距离）"""
        if self.current_scatter_data is None or len(self.current_scatter_data) == 0:
            return

        # 获取坐标轴和转换器
        ax = self.figure.axes[0]
        trans = ax.transData

        # 将点击位置转换为显示坐标
        click_display = trans.transform((x, y))

        # 计算所有点到点击位置的像素距离
        min_dist = float('inf')
        nearest_idx = -1

        for i, point in enumerate(self.current_scatter_data):
            # 将数据点转换为显示坐标
            point_display = trans.transform((point[0], point[1]))

            # 计算像素距离
            dist = np.sqrt((click_display[0] - point_display[0]) ** 2 +
                           (click_display[1] - point_display[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx >= 0:
            # 获取要删除的点坐标
            deleted_point = self.current_scatter_data[nearest_idx]

            # 删除最近的点
            self.current_scatter_data = np.delete(self.current_scatter_data, nearest_idx, axis=0)

            # 重新绘制
            self.redraw_plot()
            print(f"Point removed: ({deleted_point[0]:.2f}, {deleted_point[1]:.2f})")

    def on_motion(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata

    def redraw_plot(self):
        self.callback.emit(self.current_scatter_data)
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        # 更新散点
        if self.scatter:
            self.scatter.remove()

        if len(self.current_scatter_data) > 0:
            self.scatter = ax.scatter(self.current_scatter_data[:, 0],
                                      self.current_scatter_data[:, 1],
                                      s=10,
                                      c='red',
                                      picker=5)
        else:
            self.scatter = None

        # 仅更新画布不重建
        self.canvas.draw_idle()

    def save_plot(self, save_path):
        """保存当前绘图"""
        if self.current_data is not None:
            self.figure.savefig(save_path, bbox_inches='tight', dpi=300)

    def clear_plot(self):
        """清除绘图"""
        self.figure.clear()
        self.canvas.draw()


class InteractiveDispPlot(QWidget):
    callback = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4.38, 3.99))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.disp_temp = []
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # 创建可切换的按钮（代替复选框）
        self.toggle_interaction_btn = QPushButton("PICK")
        self.toggle_interaction_btn.setCheckable(True)  # 使按钮可切换
        self.toggle_interaction_btn.setChecked(True)  # 默认启用
        self.toggle_interaction_btn.toggled.connect(self._update_button_style)
        self.cut_toggle_btn = QPushButton("USE CUT")
        self.cut_toggle_btn.setCheckable(True)
        self.cut_toggle_btn.setChecked(True)
        self.cut_toggle_btn.toggled.connect(self._update_cut_button_style)

        # 设置按钮样式（亮/暗模式）
        self._update_button_style(self.toggle_interaction_btn.isChecked())
        self._update_cut_button_style(self.cut_toggle_btn.isChecked())

        # 添加到工具栏右侧
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.toggle_interaction_btn)
        self.toolbar.addWidget(self.cut_toggle_btn)
        # 存储当前显示的散点数据
        self.current_scatter_data = None
        self.current_pic_num = None
        self.current_params = None
        self.current_extent = None
        self.current_E_all = None

        # 连接鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def _update_button_style(self, checked):
        """根据按钮状态更新样式（亮/暗）"""
        if checked:
            # "亮"模式（启用交互）
            self.toggle_interaction_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;  /* 绿色 */
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.toggle_interaction_btn.setText("PICK ✔")
        else:
            # "暗"模式（禁用交互）
            self.toggle_interaction_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;  /* 红色 */
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            self.toggle_interaction_btn.setText("PICK ✖")

    def _update_cut_button_style(self, checked):
        if checked:
            self.cut_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1f78b4;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #176391;
                }
            """)
            self.cut_toggle_btn.setText("CUT ✔")
        else:
            self.cut_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #7f8c8d;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #6c7a7b;
                }
            """)
            self.cut_toggle_btn.setText("CUT ✖")

    def use_cut_enabled(self):
        return bool(self.cut_toggle_btn.isChecked())

    def plot(self, E_all, disp_temp_all, pic_num, params, extent):
        # try:
        """更新绘图内容"""
        self.current_E_all = E_all
        self.current_pic_num = pic_num
        self.current_params = params
        self.current_extent = extent

        # 清除前保留坐标轴范围
        prev_xlim = self.figure.axes[0].get_xlim() if self.figure.axes else None
        prev_ylim = self.figure.axes[0].get_ylim() if self.figure.axes else None

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 绘制主图像
        self.im = ax.imshow(E_all[pic_num],
                            extent=extent,
                            aspect='auto',
                            origin='lower',
                            cmap='jet')

        # 绘制散点（直接使用参数disp_temp_all）
        self.scatter = None
        step = int(params.get('step', 1))
        try:
            key = str(pic_num * step)
            scatter_data = disp_temp_all[key]
            # scatter_data = np.zeros((E_all.shape[-1], 2))
            # max_temp = np.argmax(E_all[pic_num], axis=0)
            # scatter_data[:, 0] = np.linspace(0.1, 5, len(max_temp))
            # scatter_data[:, 1] = max_temp*900/E_all.shape[1]+100
            self.callback.emit(scatter_data)
            self.current_scatter_data = scatter_data.copy()
            self.scatter = ax.scatter(scatter_data[:, 0],
                                      scatter_data[:, 1],
                                      s=10,
                                      c='white',
                                      picker=5)
        except KeyError:
            self.current_scatter_data = np.empty((0, 2))

        # # 恢复之前的视图范围
        if prev_xlim and prev_ylim:
            ax.set_xlim(prev_xlim)
            ax.set_ylim(prev_ylim)
        station_range = params.get('range', (0, len(E_all) * step))
        try:
            start_sta = int(np.ceil(float(station_range[0])))
        except (TypeError, ValueError, IndexError):
            start_sta = 0
        ax.set_xlabel('Frequency[Hz]', fontsize=12)
        ax.set_ylabel('Phase Vel.[m/s]', fontsize=12)
        ax.set_title(f'Channel {pic_num * step + start_sta} Disp')

        self.figure.tight_layout()
        self.canvas.draw()

    # except KeyError as e:
    #     QMessageBox.critical(self, '错误', f'加载文件时出错: {e}')
    #     print(f'程序出错: {e}')  # 打印错误信息到控制台

    def on_click(self, event):
        if not self.toggle_interaction_btn.isChecked():
            return
        """处理鼠标点击事件"""
        if event.inaxes:
            x, y = event.xdata, event.ydata

            # 左键添加点
            if event.button == 1:  # 左键
                self.add_point(x, y)

            # 右键删除点
            elif event.button == 3:  # 右键
                self.delete_nearest_point(x, y)

    def on_motion(self, event):
        """鼠标移动事件，用于实时显示坐标"""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            # 可以在这里添加实时坐标显示功能

    def add_point(self, x, y):

        """在指定坐标添加新点"""
        if self.current_scatter_data is None:
            self.current_scatter_data = np.empty((0, 2))

        # 添加新点
        new_point = np.array([[x, y]])
        # self.current_scatter_data = np.vstack((self.current_scatter_data, new_point))
        self.current_scatter_data = np.concatenate(
            (self.current_scatter_data, new_point),
            axis=0
        )

        # 重新绘制
        self.redraw_plot()
        print(f"Point added: ({x:.2f}, {y:.2f})")

    def delete_nearest_point(self, x, y):
        """删除距离点击位置最近的点（基于像素距离）"""
        if self.current_scatter_data is None or len(self.current_scatter_data) == 0:
            return

        # 获取坐标轴和转换器
        ax = self.figure.axes[0]
        trans = ax.transData

        # 将点击位置转换为显示坐标
        click_display = trans.transform((x, y))

        # 计算所有点到点击位置的像素距离
        min_dist = float('inf')
        nearest_idx = -1

        for i, point in enumerate(self.current_scatter_data):
            # 将数据点转换为显示坐标
            point_display = trans.transform((point[0], point[1]))

            # 计算像素距离
            dist = np.sqrt((click_display[0] - point_display[0]) ** 2 +
                           (click_display[1] - point_display[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx >= 0:
            # 获取要删除的点坐标
            deleted_point = self.current_scatter_data[nearest_idx]

            # 删除最近的点
            self.current_scatter_data = np.delete(self.current_scatter_data, nearest_idx, axis=0)

            # 重新绘制
            self.redraw_plot()
            print(f"Point removed: ({deleted_point[0]:.2f}, {deleted_point[1]:.2f})")

    def redraw_plot(self):
        self.callback.emit(self.current_scatter_data)
        if not self.figure.axes:
            return

        ax = self.figure.axes[0]

        # 更新散点
        if self.scatter:
            self.scatter.remove()

        if len(self.current_scatter_data) > 0:
            self.scatter = ax.scatter(self.current_scatter_data[:, 0],
                                      self.current_scatter_data[:, 1],
                                      s=10,
                                      c='white',
                                      picker=5)
        else:
            self.scatter = None

        # 仅更新画布不重建
        self.canvas.draw_idle()


class inv_Window(QMainWindow, inv_w):
    # image_ready = pyqtSignal(QPixmap)  # 图像就绪信号
    # image_ready2 = pyqtSignal(QPixmap)  # 图像就绪信号

    def __init__(self, para, disp_data, callback, callback2):
        super().__init__()
        self.d = None
        self.res = None
        self.count = 0
        self.thickness_all = {}
        self.layer_defs_dirty = False
        self.params = para
        if isinstance(self.params, dict):
            self.params["inv_object"] = _resolve_inv_object(self.params, default="vs")
        self._setup_async()
        # self.data1 = {}
        # self.data2 = np.zeros((len(disp_data), len(layer_definitions)))

        # self._init_matplotlib()
        self.callback2 = callback2
        self.active_threads = []
        self.running_task_ids = set()
        self.task_invert_thickness = {}
        self.setupUi(self)
        # self.setStyleSheet("""
        #      QMainWindow {
        #          background-image: url(":/jpeg/微信图片_20250307171720.png");
        #      }
        #  """)
        self.setWindowTitle(self._build_inv_window_title())
        self.model_plot = InteractiveModelPlot()
        self.verticalLayout_2.insertWidget(1, self.model_plot)
        self.model_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pro_plot = InteractiveprocessPlot()
        self.verticalLayout_3.insertWidget(1, self.pro_plot)
        self.pro_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.inv_plot = invprofileWidget()
        self.verticalLayout_4.insertWidget(3, self.inv_plot)
        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 3)
        self.verticalLayout_4.setStretch(2, 1)
        self.verticalLayout_4.setStretch(3, 3)
        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 5)
        self.inv_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.horizontalLayout_5.insertWidget(1, self.inv_plot.toolbar)
        # self.inv_plot.callback.connect(self.disp_update)
        self.disp_temp_all = dict(disp_data or {})
        filtered_dict = filter_small_arrays(self.disp_temp_all)
        new_data = interpolate_dict_to_32(filtered_dict) if filtered_dict else {}
        data_dict = {int(k): v for k, v in new_data.items()}
        new_dict = interpolate_missing_keys(data_dict) if data_dict else {}
        self._init_model_source = new_dict
        # self.disp_temp_all = new_dict
        init_model_depth = self._get_init_model_depth()
        self._last_init_model_depth = init_model_depth
        self.models = make_models(new_dict, init_model_depth)
        if self.models:
            layer_defs = convert_model_to_layer_definitions(self.models[next(iter(self.models))])
        else:
            layer_defs = self._make_empty_layer_definitions(init_model_depth)
        self.callback = callback
        self.inv_para.clicked.connect(self.para_load)
        self.layer_definitions = layer_defs
        self.d = np.asarray([float(layer['d'][0]) for layer in layer_defs], dtype=float)
        self.channel_name = list(self.disp_temp_all.keys())
        self.paramshow_2.setRowCount(len(layer_defs))
        self.paramshow_2.setHorizontalHeaderLabels(["d (km)", "Vs (m/s)", "density (kg/m³)"])
        self.runinv.clicked.connect(self.disp_inv)
        self.reinvert_btn = QPushButton("single inv")
        self.reinvert_btn.setObjectName("reinvert_current")
        self.reinvert_btn.setFont(self.runinv.font())
        self.reinvert_btn.clicked.connect(self.reinvert_selected_task)
        self.horizontalLayout.addWidget(self.reinvert_btn)
        # self.disp_scene = QGraphicsScene()
        # self.dispshow.setScene(self.disp_scene)
        # self.disp_scene2 = QGraphicsScene()
        # self.disp_scene3 = QGraphicsScene()
        if self.disp_temp_all:
            self.picchose_disp.setText('1')
        else:
            self.picchose_disp.clear()
            self.picchose_disp.setPlaceholderText("No picked dispersion data")
        self.pic_input_timer = QTimer(self)
        self.pic_input_timer.setSingleShot(True)
        self.pic_input_timer.timeout.connect(self.plot_disp_t)
        self.picchose_disp.textEdited.connect(self._schedule_disp_preview)
        self.picchose_disp.editingFinished.connect(self.plot_disp_t)
        self.picchose_disp.returnPressed.connect(self.plot_disp_t)
        self._last_manual_disp_selection = None
        self.pic_num = int(self.picchose_disp.text()) if self.picchose_disp.text() else 0
        self.res_all = np.zeros((len(self.disp_temp_all), len(self.layer_definitions)))
        self.process_plot_cache = {}
        # self.dispshow_2.setScene(self.disp_scene2)
        # self.dispshow_3.setScene(self.disp_scene3)
        self.plotp.clicked.connect(self.create_plot_canvas)
        # 填充数据
        self.populate_table()
        self.loadp.clicked.connect(self.load_res)
        self.savep.clicked.connect(self.save_res)
        # 绑定单元格修改事件
        # self.paramshow_2.cellChanged.connect(self.on_cell_changed_model)
        self.paramshow_2.cellChanged.connect(self.on_paramshow_2_cell_changed)
        self.t_list = 1 / np.linspace(para['fmin'], para['fmax'], 100)
        self.c_list = np.linspace(para['vmin'], para['vmax'], 100)
        self.pushButton.clicked.connect(self.add_row)
        self.removel.clicked.connect(self.remove_row)
        if not self.disp_temp_all:
            self.runinv.setEnabled(False)
            self.reinvert_btn.setEnabled(False)
            self.plotp.setEnabled(False)
            self.statusbar.showMessage(
                "No picked dispersion data. Load or pick dispersion curves before inversion."
            )
        # self.save_snapshot()

    def _get_init_model_depth(self):
        depth = self.params.get("init_model_depth", 80)
        try:
            depth = float(depth)
        except (TypeError, ValueError):
            depth = 80.0
        if depth <= 0:
            depth = 60.0
        return depth

    def _make_empty_layer_definitions(self, max_depth, num_layers=10):
        """Build an editable placeholder model when no picked curves are loaded."""
        num_layers = max(int(num_layers), 1)
        thickness_km = float(max_depth) / num_layers / 1000.0
        try:
            vs_min = float(self.params.get("vmin", 100.0))
            vs_max = float(self.params.get("vmax", 1000.0))
        except (TypeError, ValueError):
            vs_min, vs_max = 100.0, 1000.0
        if vs_max <= vs_min:
            vs_max = vs_min + 100.0
        return [
            {
                "d": (thickness_km, thickness_km),
                "vs": (vs_min, vs_max),
                "density": (2.0, 2.5),
            }
            for _ in range(num_layers)
        ]

    def save_snapshot(self):
        self.save_highdpi(self, _snapshot_path("inversion_window.png"), scale=10)

    def _build_inv_window_title(self):
        params = getattr(self, "params", {}) or {}
        inv_object = _resolve_inv_object(params, default="vs")
        if inv_object == "vsandd":
            mode_text = "Vs+Thickness"
        elif inv_object in {"vs_fit", "fit", "bestfit", "vsfit"}:
            mode_text = "Vs Fit"
        else:
            mode_text = "Vs Only"
        return f"inv window [{mode_text}]"

    def show_ccmessage(self, text):
        if hasattr(self, "messgeshow") and self.messgeshow is not None:
            try:
                self.messgeshow.setText(str(text))
                return
            except Exception:
                pass
        if hasattr(self, "statusbar") and self.statusbar is not None:
            self.statusbar.showMessage(str(text))
        print(text)

    def _schedule_disp_preview(self):
        if hasattr(self, "pic_input_timer"):
            self.pic_input_timer.stop()
        self._last_manual_disp_selection = None

    def _cache_digest(self, arr):
        arr = np.asarray(arr)
        h = hashlib.md5()
        h.update(str(arr.shape).encode("ascii"))
        h.update(np.ascontiguousarray(np.round(arr.astype(float), 6)).tobytes())
        return h.hexdigest()

    def _get_process_plot_cache_key(self, ls_mode):
        data_sig = self._cache_digest(self.data2) if hasattr(self, "data2") else "no-data"
        d_sig = self._cache_digest(self.d) if hasattr(self, "d") else "no-d"
        loc_sig = self._cache_digest(self.plot_loc) if hasattr(self, "plot_loc") else "no-loc"
        return (bool(ls_mode), int(self.pic_num), data_sig, d_sig, loc_sig)

    def _remember_process_plot_cache(self, key, payload):
        self.process_plot_cache[key] = payload
        if len(self.process_plot_cache) > 24:
            oldest_key = next(iter(self.process_plot_cache))
            self.process_plot_cache.pop(oldest_key, None)

    def _clone_layer_definitions(self, layer_defs):
        return [
            {
                'd': tuple(layer['d']),
                'vs': tuple(layer['vs']),
                'density': tuple(layer['density']),
            }
            for layer in layer_defs
        ]

    def save_highdpi(self, widget, filename, scale=4.0):
        size = widget.size()

        pixmap = QPixmap(size * scale)
        pixmap.setDevicePixelRatio(scale)

        painter = QPainter(pixmap)
        painter.setRenderHints(
            QPainter.Antialiasing |
            QPainter.TextAntialiasing |
            QPainter.SmoothPixmapTransform
        )

        widget.render(painter)
        painter.end()

        pixmap.save(filename)
        print("Saved:", filename)

    def populate_table(self):
        # 临时禁用信号
        self.paramshow_2.blockSignals(True)

        for row, layer in enumerate(self.layer_definitions):
            d_str = f"{layer['d'][0]}, {layer['d'][1]}"
            vs_str = f"{layer['vs'][0]}, {layer['vs'][1]}"
            density_str = f"{layer['density'][0]}, {layer['density'][1]}"

            self.paramshow_2.setItem(row, 0, QTableWidgetItem(d_str))
            self.paramshow_2.setItem(row, 1, QTableWidgetItem(vs_str))
            self.paramshow_2.setItem(row, 2, QTableWidgetItem(density_str))

        # 恢复信号
        self.paramshow_2.blockSignals(False)

    def add_row(self):
        default = {'d': (0.0, 0.0), 'vs': (0.0, 0.0), 'density': (0.0, 0.0)}
        self.layer_definitions.append(default)
        self.layer_defs_dirty = True

        new_row = self.paramshow_2.rowCount()
        self.paramshow_2.insertRow(new_row)

        d_str = f"{default['d'][0]}, {default['d'][1]}"
        vs_str = f"{default['vs'][0]}, {default['vs'][1]}"
        density_str = f"{default['density'][0]}, {default['density'][1]}"

        self.paramshow_2.setItem(new_row, 0, QTableWidgetItem(d_str))
        self.paramshow_2.setItem(new_row, 1, QTableWidgetItem(vs_str))
        self.paramshow_2.setItem(new_row, 2, QTableWidgetItem(density_str))

        self.res_all = np.zeros((len(self.disp_temp_all), len(self.layer_definitions)))

    def remove_row(self):
        if not self.layer_definitions or self.paramshow_2.rowCount() == 0:
            return

        self.layer_definitions.pop()
        self.layer_defs_dirty = True
        last_row = self.paramshow_2.rowCount() - 1
        self.paramshow_2.removeRow(last_row)

        self.res_all = np.zeros((len(self.disp_temp_all), len(self.layer_definitions)))

    def on_paramshow_2_cell_changed(self, row, column):
        column_to_key = {0: "d", 1: "vs", 2: "density"}
        key = column_to_key.get(column)
        if key is None:
            return

        item = self.paramshow_2.item(row, column)
        if item is None:
            return

        text = item.text().strip()

        try:
            parts = [v.strip() for v in text.split(',')]
            if len(parts) != 2:
                raise ValueError("Enter two comma-separated values")

            values = tuple(map(float, parts))

            self.layer_definitions[row][key] = values
            self.layer_defs_dirty = True

            print(f"Updated row {row}, field '{key}': {values}")

        except Exception as e:
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"Invalid value at row {row + 1}, column {column + 1}: {e}\nExample: 5.0, 5.0"
            )
            original_value = self.layer_definitions[row][key]
            original_str = f"{original_value[0]}, {original_value[1]}"
            self.paramshow_2.blockSignals(True)
            item.setText(original_str)
            self.paramshow_2.blockSignals(False)

    def para_load(self):
        """
        加载参数到 QTableWidget，并绑定修改更新
        """
        # self.save_snapshot()
        self.paramshow.blockSignals(True)  # 阻止信号，防止加载时触发 itemChanged

        self.paramshow.setRowCount(len(self.params))
        self.paramshow.setColumnCount(2)
        self.paramshow.setHorizontalHeaderLabels(['Parameter', 'Value'])

        for row, (key, value) in enumerate(self.params.items()):
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)  # 键不可编辑
            val_item = QTableWidgetItem(str(value))
            self.paramshow.setItem(row, 0, key_item)
            self.paramshow.setItem(row, 1, val_item)

        self.paramshow.blockSignals(False)
        self.paramshow.itemChanged.connect(self.on_param_changed)
        print("Parameter table loaded")

    def on_param_changed(self, item):
        self.paramshow.blockSignals(True)
        self.save_data()
        self.paramshow.blockSignals(False)
        # if item.column() != 1:
        #     return
        #
        # row = item.row()
        # key_item = self.paramshow.item(row, 0)
        # val_item = self.paramshow.item(row, 1)
        #
        # if not key_item or not val_item:
        #     return
        #
        # key = key_item.text().strip()
        # value_str = val_item.text().strip()
        #
        # if key not in self.params:
        #     print(f"⚠️ 未识别的参数键：{key}")
        #     return
        #
        # old_value = self.params[key]
        #
        # try:
        #     # 根据原类型尝试转换
        #     if isinstance(old_value, bool):
        #         new_value = value_str.lower() in ["true", "1", "yes", "y"]
        #     elif isinstance(old_value, int):
        #         new_value = int(value_str)
        #     elif isinstance(old_value, float):
        #         new_value = float(value_str)
        #     else:
        #         new_value = value_str
        #
        #     # 更新参数字典
        #     self.params[key] = new_value
        #
        # except Exception as e:
        #     print(f"⚠️ 参数 {key} 类型转换失败: {e}")
        #     # 恢复表格显示原来的旧值，防止非法值留在界面
        #     self.paramshow.blockSignals(True)
        #     self.save_data()
        #     val_item.setText(str(old_value))
        #     self.paramshow.blockSignals(False)
        #     return
        #
        # print(f"✅ 参数更新: {key} = {new_value}")
        # self.callback2(self.params)

    def _mark_cell_error(self, item):
        if item is not None:
            item.setBackground(QColor(255, 100, 100))  # 红色提示

    def _mark_cell_ok(self, item):
        if item is not None:
            item.setBackground(QColor(255, 255, 255))  # 恢复白色

    def convert_value(self, value_str, original_value, cell_item=None):
        """根据原类型转换，同时标红错误"""
        s = value_str.strip()

        if isinstance(original_value, bool):
            v = s.lower()
            if v in ("true", "1", "yes"):
                self._mark_cell_ok(cell_item)
                return True
            elif v in ("false", "0", "no"):
                self._mark_cell_ok(cell_item)
                return False

            self._mark_cell_error(cell_item)
            return original_value
        if isinstance(original_value, int):
            try:
                val = int(s)
                self._mark_cell_ok(cell_item)
                return val
            except:
                self._mark_cell_error(cell_item)
                return original_value

        if isinstance(original_value, float):
            try:
                val = float(s)
                self._mark_cell_ok(cell_item)
                return val
            except:
                self._mark_cell_error(cell_item)
                return original_value
        if isinstance(original_value, list):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    self._mark_cell_ok(cell_item)
                    return val
            except:
                pass
            self._mark_cell_error(cell_item)
            return original_value

        if isinstance(original_value, tuple):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, tuple):
                    self._mark_cell_ok(cell_item)
                    return val
            except:
                pass
            self._mark_cell_error(cell_item)
            return original_value

        if isinstance(original_value, np.ndarray):
            try:
                val = ast.literal_eval(s)  # 解析为 list
                arr = np.array(val)
                if isinstance(arr, np.ndarray):
                    self._mark_cell_ok(cell_item)
                    return arr
            except:
                pass

            self._mark_cell_error(cell_item)
            return original_value

        self._mark_cell_ok(cell_item)
        return s

    def _infer_type(self, s, cell_item=None):
        s = s.strip()

        # bool
        if s.lower() in ("true", "false"):
            self._mark_cell_ok(cell_item)
            return s.lower() == "true"

        # list / tuple / dict / number using literal_eval
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, dict, int, float)):
                self._mark_cell_ok(cell_item)
                return val
        except:
            pass

        # numpy array 格式 np.array([1,2,3])
        if s.startswith("np.array"):
            try:
                inner = s[s.find("(") + 1: s.rfind(")")]
                arr = np.array(ast.literal_eval(inner))
                self._mark_cell_ok(cell_item)
                return arr
            except:
                pass

        # string fallback
        self._mark_cell_ok(cell_item)
        return s

    def save_data(self):

        if self.params is None:
            QMessageBox.warning(self, "Error", "No parameters have been loaded.")
            return

        original_params = self.params
        updated_params = {}

        for row in range(self.paramshow.rowCount()):
            key_item = self.paramshow.item(row, 0)
            val_item = self.paramshow.item(row, 1)

            if not key_item or not val_item:
                continue

            key = key_item.text().strip()
            value_str = val_item.text().strip()
            original_value = original_params.get(key)

            # 传入 val_item 以便标红
            if original_value is not None:
                updated_value = self.convert_value(value_str, original_value, val_item)
            else:
                updated_value = self._infer_type(value_str, val_item)

            updated_params[key] = updated_value

        updated_params["inv_object"] = _resolve_inv_object(updated_params, default="vs")
        self.params = updated_params
        self.setWindowTitle(self._build_inv_window_title())

        print("\n======= Parameters Updated =======")
        for k, v in updated_params.items():
            print(f"{k}: {v}")
        print("=======================\n")

    def _refresh_inversion_models(self):
        self.save_data()
        init_model_depth = self._get_init_model_depth()
        depth_changed = abs(
            float(init_model_depth) - float(getattr(self, "_last_init_model_depth", init_model_depth))
        ) > 1e-9
        self.models = make_models(self._init_model_source, init_model_depth)
        self._model_keys = list(self.models.keys())
        self._last_init_model_depth = init_model_depth
        if not self.models:
            if depth_changed and not self.layer_defs_dirty:
                self.layer_definitions = self._make_empty_layer_definitions(init_model_depth)
                self.populate_table()
            print(f"[INIT MODEL] empty picked-data state, depth={init_model_depth}")
            return
        if depth_changed:
            first_model = self.models[next(iter(self.models))]
            self.layer_definitions = convert_model_to_layer_definitions(first_model)
            self.layer_defs_dirty = False
            self.populate_table()
        print(f"[INIT MODEL] using init_model_depth={init_model_depth}")

    def _resolve_task_id_from_text(self, text=None, show_error=False):
        raw_text = self.picchose_disp.text().strip() if text is None else str(text).strip()
        if not raw_text:
            if show_error:
                QMessageBox.warning(self, "Notice", "Enter an inversion index or channel number.")
            return None

        try:
            value = int(float(raw_text))
        except (TypeError, ValueError):
            if show_error:
                QMessageBox.warning(self, "Notice", f"Invalid channel input: {raw_text}")
            return None

        n_tasks = len(self.disp_temp_all)
        if 0 <= value < n_tasks:
            return int(value)

        channel_items = [str(v) for v in getattr(self, "channel_name", list(self.disp_temp_all.keys()))]
        value_str = str(value)
        if value_str in channel_items:
            return int(channel_items.index(value_str))

        if show_error:
            QMessageBox.warning(self, "Notice", f"Channel {raw_text} is outside the inversion range.")
        return None

    def _extract_target_for_task(self, task_id):
        task_id = int(task_id)
        self.channel_name = list(self.disp_temp_all.keys())
        disp_temp = np.array(list(self.disp_temp_all.values())[task_id])
        if is_all_integers_np(np.array(disp_temp)):
            target = np.argwhere(disp_temp > 0)
            target = np.hsplit(np.array(target), 2)
            target[0] = target[0] * (self.params['vmax'] - self.params['vmin']) / 256 + self.params['vmin']
            target[1] = target[1] * (self.params['fmax'] - self.params['fmin']) / 256 + self.params['fmin']
        else:
            target = np.where(disp_temp > 0, disp_temp, 0).T.copy()
            target = np.flip(target).copy()
        return target

    def _layer_defs_from_existing_result(self, task_id):
        task_id = int(task_id)
        if task_id >= len(self.res_all):
            return None

        vs_model = np.asarray(self.res_all[task_id], dtype=float)
        if vs_model.ndim != 1 or vs_model.size == 0 or not np.any(np.abs(vs_model) > 0):
            return None

        thickness = self.thickness_all.get(task_id)
        if thickness is None:
            return None
        thickness = np.asarray(thickness, dtype=float)
        if thickness.shape != vs_model.shape or not np.all(np.isfinite(thickness)):
            return None

        layer_defs = []
        for i, (vs_val, thk_val) in enumerate(zip(vs_model, thickness)):
            density = (2.0, 2.5)
            if i < len(self.layer_definitions):
                density = tuple(self.layer_definitions[i].get('density', density))
            thk_val = max(float(thk_val), 1.0e-4)
            vs_ms = max(float(vs_val) * 1000.0, 1.0)
            layer_defs.append({
                'd': (thk_val, thk_val),
                'vs': (vs_ms, vs_ms),
                'density': density,
            })
        return layer_defs

    def _build_layer_definitions_for_task(self, task_id, prefer_existing_result=False):
        if self.layer_defs_dirty:
            return self._clone_layer_definitions(self.layer_definitions)

        if prefer_existing_result:
            existing_layer_defs = self._layer_defs_from_existing_result(task_id)
            if existing_layer_defs is not None:
                return existing_layer_defs

        model_keys = getattr(self, "_model_keys", list(self.models.keys()))
        task_id = max(0, min(int(task_id), len(model_keys) - 1))
        return convert_model_to_layer_definitions(self.models[model_keys[task_id]])

    def _start_inversion_task(self, task_id, prefer_existing_result=False, manual=False):
        task_id = int(task_id)
        if task_id < 0 or task_id >= len(self.disp_temp_all):
            QMessageBox.warning(self, "Notice", f"Invalid task ID: {task_id}")
            return False

        self.clean_finished_tasks()
        if task_id in self.running_task_ids:
            channel_label = self.channel_name[task_id] if task_id < len(self.channel_name) else task_id
            self.show_ccmessage(f"Task {task_id} / Channel {channel_label} is running")
            return False

        target = self._extract_target_for_task(task_id)
        self.dispbar.setMaximum(len(self.disp_temp_all) - 1)
        self.count = 0
        inv_para = dict_to_namespace(self.params)
        task_invert_thickness = _should_invert_thickness(inv_para)
        self.task_invert_thickness[task_id] = bool(task_invert_thickness)
        layer_defs = self._build_layer_definitions_for_task(task_id, prefer_existing_result=prefer_existing_result)
        init_thickness = np.asarray([float(layer['d'][0]) for layer in layer_defs], dtype=float)
        self.thickness_all[int(task_id)] = init_thickness.copy()
        if self.d is None or int(task_id) == int(getattr(self, "pic_num", -1)):
            self.d = init_thickness.copy()

        thread = run_inv(task_id, inv_para, target, layer_defs)
        thread._inv_task_id = task_id
        thread.progress_data2.connect(
            lambda data, task_id=task_id: self.get_d(data, task_id)
        )
        thread.progress_data.connect(
            lambda data, task_id=task_id: self.plot_res(data, task_id)
        )
        thread.progress_data1.connect(
            lambda task_id=task_id: self.plot_disp_pro(task_id, target)
        )
        thread.progress_text.connect(self.show_ccmessage)
        thread.finished_signal.connect(
            lambda task_id=task_id: self.on_task_finished(task_id)
        )
        thread.start()

        self.active_threads.append(thread)
        self.running_task_ids.add(task_id)
        self.update_inv_progress(task_id)

        channel_label = self.channel_name[task_id] if task_id < len(self.channel_name) else task_id
        if manual:
            self.show_ccmessage(f"single inving: task={task_id}, channel={channel_label}")
        print(f"Started task {task_id} (channel {channel_label})")
        return True

    def check_and_start_tasks(self):
        """ 检查是否达到最大任务 ID"""
        if self.task_id >= self.max_task_id:
            print("Reached max task ID, stopping new tasks.")
            self.timer.stop()  # 停止定时器
            return
        # 检查显存是否足够
        free_memory = get_free_gpu_memory()
        if free_memory > self.memory_per_thread and len(self.active_threads) < self.max_threads:
            self.start_new_task()

        # 清理已完成的任务
        self.clean_finished_tasks()

    def clean_finished_tasks(self):
        """ 清理已完成的任务"""
        self.active_threads = [t for t in self.active_threads if t.isRunning()]
        self.running_task_ids = {
            int(getattr(t, "_inv_task_id"))
            for t in self.active_threads
            if hasattr(t, "_inv_task_id")
        }
        print(f"Active threads: {len(self.active_threads)}")

    def disp_inv(self):
        if not self.disp_temp_all:
            QMessageBox.information(
                self,
                "No picked data",
                "Load or pick dispersion curves before starting inversion.",
            )
            return
        self._refresh_inversion_models()
        self.max_threads = 8  # 鏈€澶у悓鏃惰繍琛岀殑绾跨▼鏁?
        self.memory_per_thread = 5000  # 姣忎釜绾跨▼鍗犵敤鐨勬樉瀛橈紙MB锛?
        self.max_task_id = len(self.disp_temp_all)
        self.clean_finished_tasks()
        self.task_id = 0  # 浠诲姟 ID
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_and_start_tasks)
        self.timer.start(3000)  # 姣?3 绉掓鏌ヤ竴娆?
        return
        self._refresh_inversion_models()
        self.max_threads = 8  # 最大同时运行的线程数
        self.memory_per_thread = 5000  # 每个线程占用的显存（MB）
        self.max_task_id = len(self.disp_temp_all)
        self.active_threads = []
        self.task_id = 0  # 任务 ID
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_and_start_tasks)
        self.timer.start(3000)  # 每 3 秒检查一次

    def reinvert_selected_task(self):
        if not self.disp_temp_all:
            QMessageBox.information(
                self,
                "No picked data",
                "Load or pick dispersion curves before starting inversion.",
            )
            return
        self._refresh_inversion_models()
        task_id = self._resolve_task_id_from_text(show_error=True)
        if task_id is None:
            return
        self.pic_num = int(task_id)
        self.picchose_disp.setText(str(self.pic_num))
        self._start_inversion_task(task_id, prefer_existing_result=True, manual=True)

    def start_new_task(self):
        """# 创建新任务"""
        current_task_id = self.task_id
        started = self._start_inversion_task(current_task_id, prefer_existing_result=False, manual=False)
        self.task_id += 1
        if not started:
            print(f"Skipped task {current_task_id}")
        return
        self.dispbar.setMaximum(len(self.disp_temp_all) - 1)
        disp_temp = np.array(list(self.disp_temp_all.values())[current_task_id])
        # self.disp_temp = np.sum(self.disp_temp, axis=-1)
        items = list(self.disp_temp_all.keys())
        self.channel_name = items
        if is_all_integers_np(np.array(disp_temp)):
            target = np.argwhere(disp_temp > 0)
            target = np.hsplit(np.array(target), 2)
            target[0] = target[0] * (self.params['vmax'] - self.params['vmin']) / 256 + self.params['vmin']
            target[1] = target[1] * (self.params['fmax'] - self.params['fmin']) / 256 + self.params['fmin']
        else:
            target = np.where(disp_temp > 0, disp_temp, 0).T.copy()
            target = np.flip(target).copy()
        # tat_t = np.linspace(target[0].min(), target[0].max(), 100)
        # func = interp1d(target[0], target[1], kind='linear')
        # tat_t1 = func(tat_t)
        # target = np.vstack((tat_t, tat_t1))
        self.count = 0
        inv_para = dict_to_namespace(self.params)
        # layer_defs = convert_model_to_layer_definitions(
        #     self.models[str(int(self.params['step'] * current_task_id + int(next(iter(self.models)))))])
        if self.layer_defs_dirty:
            layer_defs = self._clone_layer_definitions(self.layer_definitions)
        else:
            KEYS = list(self.models.keys())
            layer_defs = convert_model_to_layer_definitions(self.models[KEYS[current_task_id]])
        thread = run_inv(current_task_id, inv_para, target, layer_defs)
        thread.progress_data2.connect(
            lambda data, task_id=current_task_id: self.get_d(data, task_id)
        )
        thread.progress_data.connect(
            lambda data, task_id=current_task_id: self.plot_res(data, task_id)
        )
        thread.progress_data1.connect(
            lambda task_id=current_task_id: self.plot_disp_pro(task_id, target)
        )
        thread.progress_text.connect(self.show_ccmessage)
        thread.finished_signal.connect(
            lambda task_id=current_task_id: self.on_task_finished(task_id)
        )
        thread.start()

        # 记录活动线程
        self.active_threads.append(thread)
        self.update_inv_progress(current_task_id)
        self.task_id += 1
        print(f"Started task {current_task_id}")

    def get_d(self, data, task_id=None):
        allow_thickness = _should_invert_thickness(self.params)
        if task_id is not None:
            allow_thickness = self.task_invert_thickness.get(int(task_id), allow_thickness)
        if not allow_thickness:
            return
        self.d = np.asarray(data, dtype=float)
        for i, thickness in enumerate(self.d):
            if i < len(self.layer_definitions):
                old_vs = self.layer_definitions[i].get('vs', (0.0, 0.0))
                old_density = self.layer_definitions[i].get('density', (0.0, 0.0))
                self.layer_definitions[i] = {
                    'd': (float(thickness), float(thickness)),
                    'vs': old_vs,
                    'density': old_density,
                }
        if task_id is not None:
            self.thickness_all[int(task_id)] = self.d.copy()
            self.process_plot_cache = {
                k: v for k, v in self.process_plot_cache.items()
                if int(k[1]) != int(task_id)
            }
            if hasattr(self, "pic_num") and int(task_id) == int(self.pic_num):
                self.populate_table()

    def on_task_finished(self, task_id):
        # 任务完成时的回调
        print(f"Task {task_id} finished")
        self.running_task_ids.discard(int(task_id))
        self.task_invert_thickness.pop(int(task_id), None)
        self.clean_finished_tasks()

    def _setup_async(self):
        """配置异步渲染：设置线程池并连接信号"""
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool1 = QThreadPool.globalInstance()

    #     self.image_ready.connect(self._update_display)
    #     self.image_ready2.connect(self._update_display_picked)

    # def _init_matplotlib(self):
    #     # 用于主图显示
    #     self.static_fig = plt.figure(figsize=(7, 5), dpi=300)
    #     self.static_ax = self.static_fig.add_subplot(111)
    #     # 用于 picked 图显示
    #     self.static_fig2 = plt.figure(figsize=(3, 5), dpi=300)
    #     self.static_ax2 = self.static_fig2.add_subplot(111)
    #     # # 如果需要 canvas 对象可以保存
    #     self.static_canvas = self.static_fig.canvas
    #     self.static_canvas2 = self.static_fig2.canvas

    def plot_disp_t(self):
        try:
            if hasattr(self, "pic_input_timer") and self.pic_input_timer.isActive():
                self.pic_input_timer.stop()
            raw_text = self.picchose_disp.text().strip()
            resolved_task_id = self._resolve_task_id_from_text(raw_text)
            if resolved_task_id is None:
                return
            selection_token = (raw_text, int(resolved_task_id))
            if selection_token == getattr(self, "_last_manual_disp_selection", None):
                return
            self._last_manual_disp_selection = selection_token
            self.pic_num = int(resolved_task_id)
            target = self._extract_target_for_task(self.pic_num)
            self.plot_loc = target

            if self.res_all[int(self.pic_num)].any():
                self.data2 = self.res_all[int(self.pic_num)]
                self.populate_table()
                self._trigger_render()
            else:
                self._last_manual_disp_selection = None
                self.pro_plot.figure.clear()
                self.pro_plot.canvas.draw()
                if hasattr(self, 'model_plot'):
                    self.model_plot.figure.clear()
                    self.model_plot.canvas.draw()
        except Exception:
            pass

    def plot_disp_pro(self, task_id, target):
        """回调函数：接收外部传入的数据，并在满足条件时触发渲染"""
        if not self._validate_input():
            return
        if task_id == self.pic_num:
            self.plot_loc = target
            # self.generate_and_display_single_image()
            # if hasattr(self, 'data2'):
            #     self.generate_and_display_single_image_1()

            if self.count % 2 == 0:
                self._trigger_render()
            self.count += 1

    def generate_and_display_single_image_1(self):
        """# 创建 Matplotlib 图像"""
        # self.disp_scene2.clear()
        d_list = [float(layer['d'][0]) for layer in self.layer_definitions]
        x_loc = np.cumsum(d_list)
        self.model_plot.plot(x_loc, self.data2, self.channel_name[self.pic_num])

    def generate_and_display_single_image(self):
        """ 创建 Matplotlib 图像"""
        if not self.params['ls']:
            cache_key = self._get_process_plot_cache_key(ls_mode=False)
            cached = self.process_plot_cache.get(cache_key)
            if cached is not None:
                self.alpha = cached["alpha"].copy()
                self.rho = cached["rho"].copy()
                self.velocity_model2call = cached["velocity_model2call"].copy()
                self.pro_plot.plot(
                    cached["c_temp1"],
                    cached["t_temp1"],
                    cached["A"],
                    self.plot_loc,
                    int(self.channel_name[self.pic_num]),
                    draw_mode_lines=False,
                )
                return
            # self.disp_scene.clear()
            c_temp1 = np.linspace(self.params['vmin'], self.params['vmax'], 100) / 1000
            t_temp1 = 1 / np.linspace(self.params['fmin'], self.params['fmax'], 100)
            c_temp = np.tile(c_temp1, (100, 1))
            t_temp = np.tile(t_temp1, (100, 1))
            c_temp = np.reshape(c_temp, (100 * 100))
            t_temp = t_temp.T
            t_temp = np.reshape(t_temp, (100 * 100))
            c_list = numpy2tensor(c_temp)
            t_list = numpy2tensor(t_temp)
            self.beta = self.data2
            if self.params['inittal_method'] == 'Brocher':
                # 计算 alpha 和 rho 的 numpy 数组
                self.alpha = 0.9409 + 2.0947 * self.beta - 0.8206 * self.beta ** 2 + 0.2683 * self.beta ** 3 - 0.0251 * self.beta ** 4
                self.rho = 1.6612 * self.alpha - 0.4721 * self.alpha ** 2 + 0.0671 * self.alpha ** 3 - 0.0043 * self.alpha ** 4 + 0.000106 * self.alpha ** 5
            else:
                self.alpha = self.beta * 1.73
                self.rho = 0.54 * self.alpha+ 0.25
            F = _surf96_vector_gpu.dltar_vector(c_list, t_list, self.d, self.alpha, self.beta, self.rho, 2,
                                                -1, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
            A = np.array(F.cpu().reshape((100, 100)))

            self.velocity_model2call = np.vstack((self.d, self.alpha, self.beta, self.rho))
            self._remember_process_plot_cache(cache_key, {
                "alpha": np.asarray(self.alpha, dtype=float).copy(),
                "rho": np.asarray(self.rho, dtype=float).copy(),
                "velocity_model2call": np.asarray(self.velocity_model2call, dtype=float).copy(),
                "c_temp1": np.asarray(c_temp1, dtype=float).copy(),
                "t_temp1": np.asarray(t_temp1, dtype=float).copy(),
                "A": np.asarray(A, dtype=float).copy(),
            })
            self.pro_plot.plot(
                c_temp1,
                t_temp1,
                A,
                self.plot_loc,
                int(self.channel_name[self.pic_num]),
                draw_mode_lines=False,
            )
        else:
            cache_key = self._get_process_plot_cache_key(ls_mode=True)
            cached = self.process_plot_cache.get(cache_key)
            if cached is not None:
                self.alpha = cached["alpha"].copy()
                self.rho = cached["rho"].copy()
                self.pro_plot.plot(
                    cached["c_all"],
                    cached["t_list"],
                    0,
                    self.plot_loc,
                    int(self.channel_name[self.pic_num]),
                    cached["c_all"],
                    cached["t_list"],
                )
                return
            c_temp1 = np.linspace(self.params['vmin'], self.params['vmax']+500, 50)/1000
            t_temp1 = 1 / np.linspace(self.params['fmin'], self.params['fmax'], 50)
            c_temp = np.tile(c_temp1, (50, 1))
            t_temp = np.tile(t_temp1, (50, 1))
            c_temp = np.reshape(c_temp, (50 * 50))
            t_temp = t_temp.T
            t_temp = np.reshape(t_temp, (50 * 50))
            c_list = numpy2tensor(c_temp)
            t_list = numpy2tensor(t_temp)
            # 假设 self.data2 是已经存在的 numpy 数组
            self.beta = np.array(self.data2)  # 转换为 numpy 数组
            if self.params['inittal_method'] == 'Brocher':
                # 计算 alpha 和 rho 的 numpy 数组
                self.alpha = 0.9409 + 2.0947 * self.beta - 0.8206 * self.beta ** 2 + 0.2683 * self.beta ** 3 - 0.0251 * self.beta ** 4
                self.rho = 1.6612 * self.alpha - 0.4721 * self.alpha ** 2 + 0.0671 * self.alpha ** 3 - 0.0043 * self.alpha ** 4 + 0.000106 * self.alpha ** 5
            else:
                self.alpha = self.beta * 1.73
                self.rho = 0.54 * self.alpha+ 0.25
            self.d = self.d.squeeze()
            # F = _surf96_vector_gpu.dltar_vector(c_list, t_list, self.d, self.alpha, self.beta, self.rho, 2,
            #                                     -1, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
            # A = np.array(F.cpu().reshape((50, 50)))
            modes = split_dispersion_modes_2xn(self.plot_loc, max_modes=self.params.get('max_mode', 3))
            c_all = []
            _warmup_surf96_safe()
            for mode in range(len(modes)):
                c = gsurf96.surf96(
                    self.t_list,  # period array
                    self.d, self.alpha, self.beta, self.rho,
                    itype=0,  # 相速度
                    mode=mode,  # 阶次，如 0, 1, 20
                )

                c_all.append(c)
            # c1 = np.empty_like(c_temp, dtype=float)
            #
            # for i in range(len(c_temp)):
            #     c1[i] = gsurf96.dltar4(
            #         c_temp[i],
            #         t_temp[i],
            #         self.d, self.alpha, self.beta, self.rho,
            #         -1
            #     )
            # A = np.array(c1.reshape((100, 100)))
            # self.pro_plot.plot(c_temp1, t_temp1, A, self.plot_loc, int(self.channel_name[self.pic_num]),c_all, self.t_list, )
            c_all = np.asarray(c_all, dtype=float)
            t_cache = np.asarray(self.t_list, dtype=float)
            self._remember_process_plot_cache(cache_key, {
                "alpha": np.asarray(self.alpha, dtype=float).copy(),
                "rho": np.asarray(self.rho, dtype=float).copy(),
                "c_all": c_all.copy(),
                "t_list": t_cache.copy(),
            })
            self.pro_plot.plot(c_all, self.t_list, 0, self.plot_loc, int(self.channel_name[self.pic_num]), c_all, self.t_list)

    def _trigger_render(self):
        """触发渲染流程：更新显示框并调用异步渲染任务"""
        # 更新显示的台站编号
        self.picchose_disp.setText(str(self.pic_num))
        # try:
        self.plot_disp()
        # 如果存在额外数据则启动 picked 渲染任务
        if hasattr(self, 'data2') and int(self.pic_num) < len(self.res_all) and self.res_all[int(self.pic_num)].any():
            self.plot_picked()
        # except:
        #     pass

    def plot_disp(self):
        """启动显示渲染任务（异步）"""
        if not self._validate_input():
            return
        self.generate_and_display_single_image()
        task = RenderTask(self.generate_and_display_single_image)
        # self.thread_pool.start(task)

    def plot_picked(self):
        """启动 picked 渲染任务（异步）"""
        if not self._validate_input():
            return
        self.generate_and_display_single_image_1()
        task = RenderTask(self.generate_and_display_single_image_1)
        # self.thread_pool1.start(task)

    def _validate_input(self):
        """输入验证：检查台站编号输入及相关参数"""
        resolved_task_id = self._resolve_task_id_from_text(show_error=True)
        if resolved_task_id is None:
            return False
        self.pic_num = int(resolved_task_id)
        return True
        if not self.picchose_disp.text().strip():
            pass
        if self.picchose_disp.text():
            try:
                # 根据输入和步长计算 pic_num
                if self.picchose_disp.text().isdigit():
                    self.pic_num = int(self.picchose_disp.text())
                else:
                    return
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid input format.")
                return False
            return True
        else:
            pass

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        if hasattr(self, 'sa'):
            self.callback(self.sa)
        # plt.close(self.static_fig)
        super().closeEvent(event)

    def plot_res(self, data, task_id):
        self.res_all[int(task_id)] = data
        self.process_plot_cache = {
            k: v for k, v in self.process_plot_cache.items()
            if int(k[1]) != int(task_id)
        }
        if int(task_id) not in self.thickness_all and self.d is not None:
            self.thickness_all[int(task_id)] = np.asarray(self.d, dtype=float).copy()
        # self.create_plot_canvas()
        self.count += 1
        self.beta4call = self.res_all
        self.alpha4call = 0.9409 + 2.0947 * self.beta4call - 0.8206 * self.beta4call ** 2 + 0.2683 * self.beta4call ** 3 - 0.0251 * self.beta4call ** 4
        self.rho4call = 1.6612 * self.alpha4call - 0.4721 * self.alpha4call ** 2 + 0.0671 * self.alpha4call ** 3 - 0.0043 * self.alpha4call ** 4 + 0.000106 * self.alpha4call ** 5
        self.d4call = np.ones_like(self.beta4call)[:] * self.d
        velocity_model2call = np.stack((self.d4call, self.alpha4call, self.beta4call, self.rho4call)).transpose(1, 0, 2)
        self.callback(velocity_model2call)
        if task_id == self.pic_num:
            self.data2 = data
            # print(task_id)
        # print(data.shape)

    def update_inv_progress(self, value):
        # 更新进度条的值
        self.dispbar.setValue(value)

    def create_plot_canvas(self):
        # while True:
        # try:
        if not self.disp_temp_all or np.asarray(self.res_all).size == 0:
            QMessageBox.information(
                self,
                "No inversion data",
                "Run an inversion or load an inversion result before plotting the profile.",
            )
            return
        x_keys = extract_numeric_keys(self.disp_temp_all)
        if not all(isinstance(k, (int, float)) for k in x_keys):
            raise ValueError("The picked-dispersion keys must be numeric")

        x_orig = np.array(x_keys, dtype=float)
        v_matrix = np.array(self.res_all, dtype=float)  # 强制转换为浮点矩阵

        # 校验层定义数据
        if not all('d' in layer for layer in self.layer_definitions):
            raise KeyError("Every layer definition must contain a 'd' field")

        # 计算累积深度
        depths = [float(layer['d'][0]) for layer in self.layer_definitions]
        total_depth = sum(depths)
        y_orig = np.linspace(0, total_depth, v_matrix.shape[-1])

        interpolate_method = 'linear'

        # 创建插值器
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略可能的网格警告
            interp_func = RegularGridInterpolator(
                (x_orig, y_orig),
                v_matrix,
                method=interpolate_method,
                bounds_error=False,
                fill_value=np.nan
            )

        # 生成新网格（修正np.linspace参数错误）
        x_new = np.linspace(x_orig.min(), x_orig.max(), 200)
        y_new = np.linspace(y_orig.min(), y_orig.max(), 200)
        x_grid, y_grid = np.meshgrid(x_new, y_new, indexing='ij')  # 确保索引正确

        # 执行插值并处理NaN
        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        v_interp = interp_func(points)
        v_interp = np.nan_to_num(v_interp, nan=0.0)  # 将NaN替换为0
        v_interp = v_interp.reshape(x_grid.shape)
        smooth_size = int(self.params.get("inv_profile_smooth", 1))
        if smooth_size > 1:
            if smooth_size % 2 == 0:
                smooth_size += 1
            filtered = np.zeros_like(v_interp)
            for i in range(v_interp.shape[1]):
                filtered[:, i] = median_filter(v_interp[:, i], size=smooth_size, mode='reflect')
        else:
            filtered = v_interp.copy()
        self.inv_plot.plot(x_new, y_new, filtered)
        self.res_all_s = [x_new, y_new, v_interp]
        # except:
        #
        #     return
        # time.sleep(10)

    def save_res(self):
        try:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "/inv_pro_inter.pkl", "All Files (*)")
            # 打开文件并序列化写入
            with open(file_name, 'wb') as file:
                pickle.dump({
                    "grid": self.res_all_s,
                    "inv_object": _resolve_inv_object(self.params, default="vs"),
                    "thickness_all": {int(k): np.asarray(v, dtype=float) for k, v in self.thickness_all.items()},
                }, file)
            file_name1, _ = QFileDialog.getSaveFileName(self, "Save File", "/inv_res.pkl", "All Files (*)")
            # 打开文件并序列化写入
            with open(file_name1, 'wb') as file1:
                pickle.dump({
                    "vs_models": np.array(self.res_all),
                    "thickness_models": {int(k): np.asarray(v, dtype=float) for k, v in self.thickness_all.items()},
                    "inv_object": _resolve_inv_object(self.params, default="vs"),
                }, file1)
        except:
            pass

    def load_res(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load File", "/inv_res.pkl", "All Files (*)")
            # 打开文件并序列化写入
            with open(file_name, 'rb') as file:
                payload = pickle.load(file)
            if isinstance(payload, dict):
                self.res_all = np.asarray(payload.get("vs_models", self.res_all), dtype=float)
                self.thickness_all = {
                    int(k): np.asarray(v, dtype=float)
                    for k, v in payload.get("thickness_models", {}).items()
                }
                if self.thickness_all:
                    first_key = sorted(self.thickness_all)[0]
                    self.d = self.thickness_all[first_key].copy()
                if "inv_object" in payload:
                    self.params["inv_object"] = _normalize_inv_object(payload["inv_object"])
                    self.setWindowTitle(self._build_inv_window_title())
                grid = payload.get("grid")
                if grid is None:
                    return
                x_new, y_new, v_interp = grid
            else:
                x_new, y_new, v_interp = payload
            smooth_size = int(self.params.get("inv_profile_smooth", 1))
            if smooth_size > 1:
                if smooth_size % 2 == 0:
                    smooth_size += 1
                filtered = np.zeros_like(v_interp)
                for i in range(v_interp.shape[1]):
                    filtered[:, i] = median_filter(v_interp[:, i], size=smooth_size, mode='reflect')
            else:
                filtered = v_interp.copy()
            self.inv_plot.plot(x_new, y_new, filtered)
        except:
            pass


class run_inv(QThread):
    progress_data = pyqtSignal(np.ndarray, int)
    progress_data1 = pyqtSignal(int)
    progress_data2 = pyqtSignal(np.ndarray, int)
    progress_text = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, task_id, params, target, layer_definitions):
        super().__init__()
        self.params = params
        self.target = target
        self.task_id = task_id
        self.layer_definitions = layer_definitions
        self.nlayers = 5
        self.inv_object = _resolve_inv_object(params, default="vs")
        self.invert_thickness = (self.inv_object == "vsandd")

    def _make_vs_bounds(self, initial_model):
        initial_model = np.asarray(initial_model, dtype=float)
        base_lb = float(self.params.vsrange[0])
        base_ub = float(self.params.vsrange[1])

        depth_axis = np.linspace(0.0, 1.0, len(initial_model))
        lower_profile = np.clip(base_lb - 0.05 * depth_axis, 0.75, None)
        upper_profile = np.maximum(base_ub + 0.25 * depth_axis, lower_profile + 0.15)

        if len(initial_model) >= 2:
            upper_profile[-2:] += 0.08
        if len(initial_model) >= 1:
            upper_profile[-1] += 0.12

        low_bound = initial_model * lower_profile
        up_bound = initial_model * upper_profile
        deep_lower_floor = np.interp(
            depth_axis,
            [0.0, 0.35, 0.65, 1.0],
            [0.10, 0.22, 0.34, 0.45],
        )
        low_bound = np.maximum(low_bound, deep_lower_floor)
        return numpy2list(low_bound), numpy2list(up_bound)

    def _make_thickness_bounds(self, initial_thickness):
        initial_thickness = np.asarray(initial_thickness, dtype=float)
        low_bound = []
        up_bound = []

        for i, layer in enumerate(self.layer_definitions):
            d_pair = layer.get('d', (initial_thickness[i], initial_thickness[i]))
            d0 = float(initial_thickness[i])
            d_min = float(min(d_pair))
            d_max = float(max(d_pair))

            if abs(d_max - d_min) < 1e-10:
                d_min = max(d0 * 0.70, 1.0e-4)
                d_max = max(d0 * 1.30, d_min + 1.0e-4)
            else:
                d_min = max(d_min, 1.0e-4)
                d_max = max(d_max, d_min + 1.0e-4)

            low_bound.append(d_min)
            up_bound.append(d_max)

        return numpy2list(low_bound), numpy2list(up_bound)

    def _resample_dispersion_curve(self, t_raw, c_raw):
        t_raw = np.asarray(t_raw, dtype=float)
        c_raw = np.asarray(c_raw, dtype=float)
        n_points = max(8, int(getattr(self.params, "disp_resample_points", 32)))
        resample_mode = str(getattr(self.params, "disp_resample_mode", "logf")).strip().lower()

        if t_raw.size <= 1:
            return t_raw.copy(), c_raw.copy()

        f_raw = 1.0 / np.maximum(t_raw, 1.0e-8)
        sort_idx = np.argsort(f_raw)
        f_sorted = f_raw[sort_idx]
        c_sorted = c_raw[sort_idx]

        if resample_mode in {"linear_period", "period", "t"}:
            t_new = np.linspace(float(np.min(t_raw)), float(np.max(t_raw)), n_points)
            interp_fun = interp1d(t_raw, c_raw, kind='linear', bounds_error=True)
            c_new = interp_fun(t_new)
            return np.asarray(t_new, dtype=float), np.asarray(c_new, dtype=float)

        if float(np.min(f_sorted)) <= 0.0:
            f_new = np.linspace(float(np.min(f_sorted)), float(np.max(f_sorted)), n_points)
        elif resample_mode in {"linear_f", "frequency", "f"}:
            f_new = np.linspace(float(np.min(f_sorted)), float(np.max(f_sorted)), n_points)
        else:
            f_new = np.geomspace(float(np.max(f_sorted)), float(np.min(f_sorted)), n_points)[::-1]

        interp_fun = interp1d(f_sorted, c_sorted, kind='linear', bounds_error=True)
        c_new = interp_fun(f_new)
        t_new = 1.0 / np.maximum(f_new, 1.0e-8)
        period_idx = np.argsort(t_new)
        return np.asarray(t_new[period_idx], dtype=float), np.asarray(c_new[period_idx], dtype=float)

    def run(self):
        # try:
        inv_object = self.inv_object
        invert_thickness = self.invert_thickness
        mode_info = (
            f"Task {self.task_id}: inv_object={inv_object}, "
            f"invert_thickness={invert_thickness}"
        )
        print(mode_info)
        self.progress_text.emit(mode_info)
        if not self.params.ls:
            callback_plotdata = self.progress_data.emit
            callback_plotdata1 = self.progress_data1.emit
            callback_plotdata2 = self.progress_data2.emit if invert_thickness else (lambda *args, **kwargs: None)
            print(f"Task {self.task_id} started")
            INV = grad_cal(
                self.params,
                self.target,
                AK135_data=[],
                device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
            )
            # 实际调用
            INV.inv_pro(
                self.layer_definitions,
                callback_plotdata,
                callback_plotdata1,
                callback_plotdata2,
                self.task_id
            )

            print(f"Task {self.task_id} completed")
            self.finished_signal.emit(self.task_id)  # 任务完成，发送信号
            # except Exception as e:
            #     print(f"ERROR in thread: {str(e)}")
        else:
            callback_plotdata = self.progress_data.emit
            callback_plotdata1 = self.progress_data1.emit
            callback_plotdata2 = self.progress_data2.emit if invert_thickness else (lambda *args, **kwargs: None)
            d, initial_model, _ = map(list, zip(*[(layer['d'], layer['vs'], layer['density']) for layer in
                                                  self.layer_definitions]))
            self.t = {}
            self.c = {}
            self.d = np.array(d)[:, 0]
            if invert_thickness:
                callback_plotdata2(self.d, self.task_id)
            self.initial_model = np.mean(np.array(initial_model), axis=-1) / 1000
            max_mode = getattr(self.params, "max_mode", 3)
            all_targets = split_dispersion_modes_2xn(self.target, max_modes=None)
            targets = split_dispersion_modes_2xn(self.target, max_modes=max_mode)
            detected_modes = len(all_targets)
            used_modes = len(targets)
            mode_msg = (
                f"Task {self.task_id}: detected {detected_modes} mode(s), "
                f"using {used_modes} mode(s) for inversion (max_mode={max_mode})"
            )
            print(mode_msg)
            self.progress_text.emit(mode_msg)
            for split_idx, mode_arr in targets.items():
                c_dbg, f_dbg = mode_arr
                c_dbg = np.asarray(c_dbg, dtype=float)
                f_dbg = np.asarray(f_dbg, dtype=float)
                inv_mode_id = len(targets) - int(split_idx) - 1
                dbg_msg = (
                    f"Task {self.task_id}: split_mode[{int(split_idx)}] -> inv_mode[{inv_mode_id}], "
                    f"points={len(c_dbg)}, "
                    f"f=[{float(np.min(f_dbg)):.3f}, {float(np.max(f_dbg)):.3f}] Hz, "
                    f"v_median={float(np.median(c_dbg)):.3f} m/s"
                )
                print(dbg_msg)
                self.progress_text.emit(dbg_msg)
            for i in range(len(targets)):
                c_raw, t_raw = targets[i]

                t_raw = 1 / np.array(t_raw)  # 周期
                c_raw = np.array(c_raw) / 1000  # km/s

                # 排序（防止顺序混乱）
                idx = np.argsort(t_raw)
                t_raw = t_raw[idx]
                c_raw = c_raw[idx]

                # 新的周期采样（加密 3 倍）
                t_new, _ = self._resample_dispersion_curve(t_raw, c_raw)

                # 插值
                f_interp = interp1d(
                    t_raw,
                    c_raw,
                    kind='linear',  # 或 'cubic'
                    bounds_error=True  # 禁止外推
                )

                c_new = f_interp(t_new)

                # 存入反演结构
                self.t[len(targets)-i-1] = t_new
                self.c[len(targets)-i-1] = c_new
                mapped_msg = (
                    f"Task {self.task_id}: inv_mode[{len(targets)-i-1}] "
                    f"uses split_mode[{i}] after resample, "
                    f"points={len(c_new)}, "
                    f"T=[{float(np.min(t_new)):.4f}, {float(np.max(t_new)):.4f}] s, "
                    f"V=[{float(np.min(c_new) * 1000.0):.3f}, {float(np.max(c_new) * 1000.0):.3f}] m/s"
                )
                print(mapped_msg)
                self.progress_text.emit(mapped_msg)
            low_bound, ub_bound = self._make_vs_bounds(self.initial_model)
            d_low_bound, d_up_bound = self._make_thickness_bounds(self.d)
            _warmup_surf96_safe()
            # _ = invert_surf96(self.initial_model, self.d, self.t, self.c, low_bound, ub_bound, callback_plotdata,
            #                   callback_plotdata1, self.task_id)
            # best = invert_surf96_multistart(
            #     self.initial_model,
            #     self.d,
            #     self.t,
            #     self.c,
            #     low_bound,
            #     ub_bound,
            #     callback_plotdata,
            #     callback_plotdata1,
            #     self.task_id,
            #     nstart=50,  # ★ MC 次数
            # )
            best = invert_surf96_adaptive_mc(
                self.initial_model,
                self.d,
                self.t,
                self.c,
                low_bound,
                ub_bound,
                callback_plotdata,
                callback_plotdata1,
                self.task_id,
                inv_object=inv_object,
                d_low_bound=d_low_bound,
                d_up_bound=d_up_bound,
                nstart=self.params.step_size,  # 每一轮 MC 数
                n_round=getattr(self.params, "mc_rounds", self.params.step),
                init_scale=self.params.lr,  # 初始扰动幅度（±30%）
                scale_decay=self.params.gamma,  # 每轮缩小
            )
            if isinstance(best, dict):
                if not invert_thickness:
                    best["thickness"] = None
                if invert_thickness and best.get("thickness") is not None:
                    old_d = np.asarray(self.d, dtype=float).copy()
                    self.d = np.asarray(best["thickness"], dtype=float)
                    delta_d = self.d - old_d
                    thickness_msg = (
                        f"Task {self.task_id}: thickness update\n"
                        f"initial={np.array2string(old_d, precision=4)}\n"
                        f"final={np.array2string(self.d, precision=4)}\n"
                        f"delta={np.array2string(delta_d, precision=4)}"
                    )
                    print(thickness_msg)
                    self.progress_text.emit(thickness_msg)
                    callback_plotdata2(self.d, self.task_id)
                if best.get("model") is not None:
                    callback_plotdata(np.asarray(best["model"], dtype=float), self.task_id)
                    callback_plotdata1(self.task_id)
            self.finished_signal.emit(self.task_id)

            # invert_surf96(initial_model,initial_d, nlayers, t_dict, obs_dict)
            
class RenderTask(QRunnable):
    """异步渲染任务"""

    def __init__(self, render_func):
        super().__init__()
        self.render_func = render_func

    def run(self):
        self.render_func()


class pick_Window(QMainWindow, pick_w):
    image_ready = pyqtSignal(QPixmap)  # 图像就绪信号
    image_ready2 = pyqtSignal(QPixmap)  # 图像就绪信号

    def __init__(self, params, disp_all, E_allextent, callback):
        super().__init__()
        self.setupUi(self)  # 假设使用Qt Designer生成的UI类
        # self.setStyleSheet("""
        #      QMainWindow {
        #          background-image: url(":/jpeg/微信图片_20250307171720.png");
        #      }
        #  """)
        self._init_ui(params, disp_all, E_allextent, callback)
        self.data = []
        self.pick_but.clicked.connect(self.pick_func)
        # self.save_snapshot()

    def save_snapshot(self):
        self.save_highdpi(self, _snapshot_path("picking_window.png"), scale=10)

    def save_highdpi(self, widget, filename, scale=4.0):
        size = widget.size()

        pixmap = QPixmap(size * scale)
        pixmap.setDevicePixelRatio(scale)

        painter = QPainter(pixmap)
        painter.setRenderHints(
            QPainter.Antialiasing |
            QPainter.TextAntialiasing |
            QPainter.SmoothPixmapTransform
        )

        widget.render(painter)
        painter.end()

        pixmap.save(filename)
        print("Saved:", filename)

    def _init_ui(self, params, disp_all, E_allextent, callback):
        """界面初始化"""
        self.params = dict(params or {})
        step = max(int(self.params.get("step", 1)), 1)
        try:
            image_count = int(np.asarray(E_allextent[0]).shape[0])
        except Exception:
            image_count = 0
        self.params.setdefault("range", [0, image_count * step])
        self.callback = callback
        self.disp_all = disp_all
        self.E_allextent = E_allextent
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.parampath.setPlaceholderText("Select UNet weights (.pth or .pt)...")
        model_path = str(params.get("picker_model_path", DEFAULT_MODEL_PATH))
        if os.path.isfile(model_path):
            self.parampath.setText(model_path)
        old_left = self.dispshow
        self.dispshow = InteractiveDispPlot(self)
        self.verticalLayout.replaceWidget(old_left, self.dispshow)
        old_left.deleteLater()

        old_right = self.dispshow_2
        self.dispshow_2 = InteractiveDispPlot(self)
        self.verticalLayout_2.replaceWidget(old_right, self.dispshow_2)
        old_right.deleteLater()
        for widget in (self.dispshow, self.dispshow_2):
            widget.toggle_interaction_btn.hide()
            widget.cut_toggle_btn.hide()

        # 连接信号槽
        self.plotdisp.clicked.connect(self.plot_disp)
        self.pushButton_3.clicked.connect(self.select_file)
        self.picchose_disp.setPlaceholderText('sta')
        self.last_disp.clicked.connect(self.turn_last_disp)
        self.paramsload.clicked.connect(self.model_load)
        self.next_disp.clicked.connect(self.turn_next_disp)

    def pick_func(self):
        self.E_all, self.extent = self.E_allextent
        self.data_resize()
        foreground_probabilities = []
        class_probabilities = []
        self.net.eval()

        for image in tqdm.tqdm(self.data_loader):

            # print("标签种类：", segment_image.max(), segment_image.min())
            # print(image.size)
            # print(segment_image.size)
            image = image.permute(0, 1, 3, 2)
            image = image.to(self.device)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            with torch.no_grad():
                logits = self.net(image)
                probabilities = torch.softmax(logits, dim=1)
            class_probabilities.append(probabilities[0, 1:].cpu().numpy().transpose(0, 2, 1))
            foreground = probabilities[:, 1:, :, :].sum(dim=1)
            foreground_probabilities.append(foreground[0].cpu().numpy().T)
        """
                self.inputtensor[:, np.newaxis, :, :, :]  # 形状 (50, 1, 256, 256, 3)
                * out_image[:, 1, :, :, np.newaxis]  # 形状 (50, 5, 256, 256, 1)
        )
        # sa[sa < 0] = 0
        # sa[sa > 0] = 1
        self.sa = torch.sum(sa[:, 1, :, :, :], axis=1).squeeze()
        self.sa = self.sa.permute(0, 2, 1, 3)
        """
        self.pick_class_probabilities = np.asarray(class_probabilities, dtype=np.float32)
        self.pick_probabilities = np.asarray(foreground_probabilities, dtype=np.float32)
        self.centerline_curves = []
        self.centerline_points = []
        masks = []
        fmin, fmax, vmin, vmax = np.asarray(self.extent, dtype=float)
        picker_extent = (vmin, vmax, fmin, fmax)
        for probability in self.pick_probabilities:
            picker_curves, mask = extract_dispersion_centerlines(
                probability,
                picker_extent,
                threshold=float(self.params.get("pick_threshold", 0.45)),
                min_area=int(self.params.get("pick_min_area", 60)),
                min_width=int(self.params.get("pick_min_width", 16)),
                max_gap=int(self.params.get("pick_max_gap", 4)),
                smooth_sigma=float(self.params.get("pick_smooth_sigma", 1.5)),
                sample_step=int(self.params.get("pick_sample_step", 2)),
            )
            curves = [curve[:, [1, 0]] for curve in picker_curves]
            self.centerline_curves.append(curves)
            self.centerline_points.append(combine_centerlines(curves))
            masks.append(mask)
        self.sa = np.asarray(masks, dtype=np.uint8)

        self.plot_picked()

        self.callback(self.centerline_points)

    def data_resize(self):
        data = self.E_all
        # 1. 数据归一化到 [0, 1]
        # normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # 输入数据形状：(n, H, W, C)
        # 混合精度计算（节省显存）
        if data.dtype == np.float32:
            min_vals = np.min(data, axis=(1, 2), keepdims=True).astype(np.float16)
            max_vals = np.max(data, axis=(1, 2), keepdims=True).astype(np.float16)
        else:
            min_vals = np.min(data, axis=(1, 2), keepdims=True)  # 沿空间维度计算
            max_vals = np.max(data, axis=(1, 2), keepdims=True)
        epsilon = 1e-8
        normalized_data = np.clip((data - min_vals) / (max_vals - min_vals + epsilon), 0, 1)
        # normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-8)  # 防止除零
        cmap = plt.cm.get_cmap('viridis')
        rgb_data = cmap(normalized_data)[:, :, :, :3]  # 忽略Alpha通道，得到形状为 (H, W, 3)

        rgb_data_uint8 = (rgb_data * 255).astype(np.uint8)
        E_tensor = torch.tensor(rgb_data_uint8, dtype=torch.float) / 255
        original_tensor = E_tensor.permute(0, 3, 2, 1)

        # 使用双线性插值
        resized_tensor = F.interpolate(
            original_tensor,
            size=(256, 256),
            mode="bilinear",  # 或 "bicubic"
            align_corners=False
        )
        self.inputtensor = resized_tensor.permute(0, 3, 2, 1)
        # self.inputtensor = torch.flip(self.inputtensor, dims=[1])
        # resized_tensor = resized_tensor.int()
        # self.resized_tensor = resized_tensor.permute(0, 2, 3, 1)
        self.data_loader = DataLoader(resized_tensor, batch_size=1, shuffle=False)

    def model_load(self):
        self.model_path = self.parampath.text()
        self.net = UNet(6).to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.net.load_state_dict(state)
        QMessageBox.information(self, "Notice", "Model loaded successfully.")

    def _setup_async(self):
        """异步渲染配置"""
        self.thread_pool = QThreadPool.globalInstance()
        self.image_ready.connect(self._update_display)
        self.thread_pool2 = QThreadPool.globalInstance()
        self.image_ready2.connect(self._update_display_picked)

    def _init_matplotlib(self):
        """预创建Matplotlib资源"""
        self.static_fig = plt.figure()
        self.static_ax = self.static_fig.add_subplot(111)
        self.static_fig2 = plt.figure()
        self.static_ax2 = self.static_fig2.add_subplot(111)
        self.static_canvas = self.static_fig.canvas
        self.static_canvas2 = self.static_fig2.canvas

    def select_file(self):
        """文件选择（保持原逻辑）"""
        self.model_path, _ = QFileDialog.getOpenFileName(
            self, 'Select File', '', 'PyTorch Models (*.pth *.pt *.ckpt);;All Files (*)')
        if self.model_path:
            self.parampath.setText(self.model_path)

    def turn_last_disp(self):
        """上一页按钮"""
        if not hasattr(self, 'pic_num'):
            self.pic_num = 0
        self.pic_num = max(0, self.pic_num - 1)  # 最小值为0
        self.save_snapshot()
        self._trigger_render()

    def turn_next_disp(self):
        """下一页按钮"""
        if not hasattr(self, 'pic_num'):
            self.pic_num = -1
        max_num = len(self.E_all) - 1
        self.pic_num = min(max_num, self.pic_num + 1)  # 最大值为数据长度-1
        self._trigger_render()

    def _trigger_render(self):
        """触发渲染流程"""
        self.picchose_disp.setText(str(self.pic_num * self.params['step']))  # 同步显示框
        self.plot_disp()  # 调用渲染方法

        if hasattr(self, 'sa'):
            self.plot_picked()

    def plot_disp(self):
        """更新后的绘图入口"""
        if not self._validate_input():
            return
        self.dispshow.plot(self.E_all, {}, self.pic_num, self.params, self.extent)

    def plot_picked(self):
        """更新后的绘图入口"""
        if not self._validate_input():
            return
        key = str(self.pic_num * int(self.params['step']))
        picked = {key: self.centerline_points[self.pic_num]}
        self.dispshow_2.plot(self.E_all, picked, self.pic_num, self.params, self.extent)

    def _validate_input(self):
        """输入验证"""
        if not self.picchose_disp.text().strip():
            QMessageBox.warning(self, "Notice", "Enter the channel number to display.")
            return False
        try:
            self.pic_num = int(int(self.picchose_disp.text()) / int(self.params['step']))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input format.")
            return False
        return True

    def _show_loading(self):
        """显示加载提示"""
        self.disp_scene.clear()
        self.loading_label = QLabel("Loading...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 24px; color: #666;")
        self.disp_scene.addWidget(self.loading_label)

    def _show_loading_picked(self):
        """显示加载提示"""
        self.disp_scene2.clear()
        self.loading_label = QLabel("Loading...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 24px; color: #666;")
        self.disp_scene2.addWidget(self.loading_label)

    def _async_render(self, pic_num):
        """后台渲染任务"""
        try:
            # Matplotlib绘图操作
            self.static_ax.clear()
            img = self.static_ax.imshow(
                self.E_all[pic_num],
                extent=self.extent,
                aspect='auto',
                origin='lower',
                cmap='jet'
            )

            # 配置坐标轴
            self.static_ax.set_xlabel('Frequency[Hz]', fontsize=12)
            self.static_ax.set_ylabel('Phase Vel.[m/s]', fontsize=12)
            self.static_ax.set_xlim(2, 35)
            self.static_ax.set_ylim(150, 760)
            self.static_ax.xaxis.set_major_locator(MultipleLocator(5))
            self.static_ax.yaxis.set_major_locator(MultipleLocator(100))
            self.static_ax.xaxis.set_minor_locator(MultipleLocator(2.5))
            self.static_ax.yaxis.set_minor_locator(MultipleLocator(50))
            self.static_ax.tick_params(which='major', length=8, width=2.0, direction='out')
            self.static_ax.tick_params(which='minor', length=3, width=0.8, direction='out')
            start_channel = int(np.asarray(self.params.get('range', [0]))[0])
            channel = start_channel + self.pic_num * int(self.params["step"])
            self.static_ax.set_title(f'Channel {channel} Disp')

            # 渲染到缓冲区
            buf = io.BytesIO()
            self.static_fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # 生成QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            self.image_ready.emit(pixmap)
        except Exception as e:
            print(f"Error: {str(e)}")

    def _async_render_picked(self, pic_num):
        """后台渲染任务"""
        # try:
        # Matplotlib绘图操作
        self.static_ax2.clear()
        # 关键修改1：移除已存在的colorbar
        if hasattr(self, 'colorbar') and self.colorbar:
            self.colorbar.remove()  # 移除旧的colorbar
        img = self.static_ax2.imshow(
            self.E_all[pic_num],
            extent=self.extent,
            aspect='auto',
            origin='lower',
            cmap='jet',
        )
        display_styles = [
            ("Fundamental mode", "#2ca02c"),
            ("Mode 1", "#1f77b4"),
            ("Mode 2", "#d62728"),
        ]
        curve_order = order_centerlines_by_min_frequency(self.centerline_curves[pic_num])
        style_by_curve = {}
        for rank, curve_index in enumerate(curve_order):
            style_by_curve[curve_index] = (
                display_styles[rank]
                if rank < len(display_styles)
                else ("Unknown mode", "#ff7f0e")
            )
        mode_handles = {}
        for mode_id, curve in enumerate(self.centerline_curves[pic_num]):
            mode_label, mode_color = style_by_curve[mode_id]
            line, = self.static_ax2.plot(
                curve[:, 0],
                curve[:, 1],
                color=mode_color,
                linewidth=2.0,
                label=mode_label,
            )
            mode_handles[mode_id] = line
        if mode_handles:
            self.static_ax2.legend(
                handles=[mode_handles[i] for i in curve_order],
                loc="best",
                fontsize=8,
            )
        # # 关键修改2：将colorbar保存为实例变量
        # self.colorbar = self.static_ax2.figure.colorbar(  # 保存到self.colorbar
        #     img,
        #     ax=self.static_ax2,
        #     orientation='vertical'
        # )
        # self.colorbar.set_label('Amplitude', fontsize=12)
        # # 配置坐标轴
        self.static_ax2.set_xlabel('Frequency[Hz]', fontsize=12)
        self.static_ax2.set_ylabel('Phase Vel.[m/s]', fontsize=12)
        self.static_ax2.set_xlim(2, 35)
        self.static_ax2.set_ylim(150, 760)
        self.static_ax2.xaxis.set_major_locator(MultipleLocator(5))
        self.static_ax2.yaxis.set_major_locator(MultipleLocator(100))
        self.static_ax2.xaxis.set_minor_locator(MultipleLocator(2.5))
        self.static_ax2.yaxis.set_minor_locator(MultipleLocator(50))
        self.static_ax2.tick_params(which='major', length=8, width=2.0, direction='out')
        self.static_ax2.tick_params(which='minor', length=3, width=0.8, direction='out')
        start_channel = int(np.asarray(self.params.get('range', [0]))[0])
        channel = start_channel + self.pic_num * int(self.params["step"])
        self.static_ax2.set_title(f'Channel {channel} Disp_picked')

        # 渲染到缓冲区
        buf = io.BytesIO()
        self.static_fig2.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        # 生成QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.image_ready2.emit(pixmap)
        # except Exception as e:
        #     print(f"错误: {str(e)}")

    def _update_display(self, pixmap):
        """更新显示（主线程执行）"""
        self.disp_scene.clear()
        scaled_pixmap = pixmap.scaled(
            self.dispshow.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        scaled_pixmap.setDevicePixelRatio(self.devicePixelRatio())
        self.disp_scene.addPixmap(scaled_pixmap)
        self.dispshow.fitInView(self.disp_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _update_display_picked(self, pixmap):
        """更新显示（主线程执行）"""
        self.disp_scene2.clear()
        scaled_pixmap = pixmap.scaled(
            self.dispshow.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        scaled_pixmap.setDevicePixelRatio(self.devicePixelRatio())
        self.disp_scene2.addPixmap(scaled_pixmap)
        self.dispshow_2.fitInView(self.disp_scene2.itemsBoundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        if hasattr(self, 'centerline_points'):
            self.callback(self.centerline_points)
        super().closeEvent(event)


class LTSC_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Window")
        self.setGeometry(300, 300, 250, 150)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("New window", self))
        self.setLayout(layout)


class run_cc(QThread):
    progress_signal = pyqtSignal(int)  # 进度信号
    progress_data = pyqtSignal(np.ndarray)  # 数据信号
    progress_str1 = pyqtSignal(str)
    # progress_str2 = pyqtSignal(str)
    progress_str2 = pyqtSignal(UTCDateTime)
    progress_str3 = pyqtSignal(UTCDateTime)
    finished_signal = pyqtSignal(str)  # 完成信号

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_running = True

    def run(self):
        callback_func = self.progress_signal.emit
        callback_plotdata = self.progress_data.emit
        callback_str1 = self.progress_str1.emit
        callback_str2 = self.progress_str2.emit
        callback_str3 = self.progress_str3.emit
        cut_cal_data(self.params, callback_func, callback_plotdata, callback_str1, callback_str2, callback_str3,
                     stop_flag=lambda: self._is_running)
        self.progress_signal.emit(100)
        self.finished_signal.emit('Work Done!')

    def stop(self):
        self._is_running = False
        print(self._is_running)
        self.quit()
        # self.wait()


class run_disp(QThread):
    progress_signal = pyqtSignal(int)  # 进度信号
    progress_data = pyqtSignal(list)  # 数据信号
    # progress_data1 = pyqtSignal(list)  # 数据信号
    progress_str1 = pyqtSignal(str)
    # progress_str2 = pyqtSignal(str)
    finished_signal = pyqtSignal(str)  # 完成信号
    task_finished = pyqtSignal()

    def __init__(self, params, cc_Data):
        super().__init__()
        self.params = params
        self.ccdata = cc_Data
        # self.ccdata = np.zeros(cc_Data.shape[0], 100, cc_Data.shape[-1])
        # for i in range(self.ccdata):
        #     sta = int(i*int(params['step']))
        #     if sta + 101 > cc_Data.shape[-2]:
        #         self.ccdata[i] = cc_Data[i, sta -101:sta-1, :]
        #     self.ccdata[i] = cc_Data[i, sta:sta+100, :]

    def run(self):
        callback_func = self.progress_signal.emit
        callback_plotdata = self.progress_data.emit
        callback_str1 = self.progress_str1.emit
        # callback_plickdata = self.progress_data1.emit
        try:
            if self.params['method'] == 'Phaseshift':
                disp_cal.cal(self.params, self.ccdata, callback_func, callback_plotdata, callback_str1)
            elif self.params['method'] == 'FK':
                disp_cal.cal_fk(self.params, self.ccdata, callback_func, callback_plotdata, callback_str1)
            elif self.params['method'] == 'cc_fj':
                disp_cal.cc_fj(self.params, self.ccdata, callback_func, callback_plotdata, callback_str1)
            else:
                raise ValueError('This method is not finished')
        except Exception as exc:
            traceback.print_exc()
            self.finished_signal.emit(f'Dispersion calculation failed: {exc}')
        else:
            self.finished_signal.emit('Work Done!')
        finally:
            self.task_finished.emit()


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))
        # 仍然保持原始stdout功能（可选）
        sys.__stdout__.write(text)


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        # validator = LicenseValidator("public_key.pem")

        self.vel_model = None
        self.FK_DATA = None
        # validator = LicenseValidator(PUBLIC_KEY)
        # result = validator.validate_license("new_license.lic")
        # if result["valid"]:
        #     print(f"✅ 许可证验证通过")
        #     print(f"用户ID: {result['user_id']}")
        #     print(f"有效期至: {result['expiry_date']}")
        #     QMessageBox.warning(self, "提示", "License is valid. Starting program...")
        # else:
        #     print(f"❌ 许可证无效: {result['error']}")
        #     return

        self.params_temp = {}
        self.cc_file_params = {}
        self.cc_file_path = None
        self.first_call = True
        self.INV = None
        self.params_inv = None
        self.target = None
        self.E_allextent = None
        self.params = None  # 存储从 YAML 文件中加载的参数
        self.file_path = None  # 存储文件路径
        self.x_cut = []
        self.x_cut_fk = []
        self.cc_data = []
        self.E_allextent = []
        self.disp_temp_all = {}
        self.setupUi(self)
        self.action.triggered.connect(self.LTSC)
        # self.setStyleSheet("""
        #      QMainWindow {
        #          background-image: url(":/jpeg/微信图片_20250307171720.png");
        #      }
        #  """)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.verticalLayout_2.insertWidget(6, self.text_edit)
        self.text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sys.stdout = EmittingStream(textWritten=self.handle_output)
        sys.stderr = EmittingStream(textWritten=self.handle_output)
        # params = self.params_all['cc']
        # cc_test = read_data(params['example_path'], para=False, data_temp=True)
        # self.cc_test = cc_test[np.newaxis,:,:]
        self.ccbar.setMinimum(0)
        self.disp_plot = InteractiveDispPlot()
        self.verticalLayout_3.insertWidget(1, self.disp_plot)
        self.disp_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.disp_plot.callback.connect(self.disp_update)
        self.free_plot = InteractiveFreePlot()
        self.verticalLayout_3.insertWidget(4, self.free_plot)
        self.free_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.free_plot.callback.connect(self.FKdata)
        self.free_plot.callback_cut.connect(self.fk_cut_save_func)
        self.free_plot.callback_cut_processed.connect(self.fk_cut_processed_update)
        self.cc_plot = RawNCFPlotWidget()
        self.verticalLayout_4.insertWidget(1, self.cc_plot)
        self.cc_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cc_plot.callback.connect(self.cut_save_func)
        # self.cc_plot.callback1.connect(self.FKdata_update)
        # 连接信号与槽
        # self.cc_data = np.zeros((50, 256, 256))
        self.parampath.setPlaceholderText('Select a file...')  # 设置输入框占位符
        if os.path.isfile(DEFAULT_CONFIG_PATH):
            self.parampath.setText(DEFAULT_CONFIG_PATH)
        self.pushButton_3.clicked.connect(self.select_file)  # 绑定文件选择按钮
        self.paramsload.clicked.connect(self.load_params)  # 绑定加载参数按钮
        # self.savecc.clicked.connect(self.save_data)  # 绑定保存按钮
        self.ccrun_2.clicked.connect(self.FKdata)
        self.ccstop.clicked.connect(self.stop_cc)
        self.rundisp.clicked.connect(self.start_run_disp)
        self.plotbut.clicked.connect(lambda: self.plot_with_matplotlib(self.cc_data))
        self.plotdisp.clicked.connect(lambda: self.plot_disp(self.E_allextent))
        self.ccrun.clicked.connect(self.start_run_cc)
        self.picchose.setPlaceholderText('sta')
        self.picchose_disp.setPlaceholderText('sta')
        self.sen_pick.setPlaceholderText('Sen_sta')
        # Keep only one draw button + station text box for sensitivity.
        # The analysis type is selected by free_plot.analysis_combo.
        self.ccrun_2.setText("Plot")
        try:
            self.pushButton.hide()
        except Exception:
            pass
        self.cut_save.clicked.connect(self.line_to_inter)
        self.paramloaddisp.clicked.connect(self.load_params_disp)
        self.savecc.clicked.connect(self.save_ccdata)
        self.pick.clicked.connect(self.open_pick_Window)
        self.inv_para.clicked.connect(self.open_inv_Window)
        self.last_cc.clicked.connect(self.turn_last_cc)
        self.next_cc.clicked.connect(self.turn_next_cc)
        self.last_disp.clicked.connect(self.turn_last_disp)
        self.next_disp.clicked.connect(self.turn_next_disp)
        self.ccload.clicked.connect(self.load_cc)
        self.savedisp.clicked.connect(self.savedisp_picdata)
        self.save_disp.clicked.connect(self.savedisp_pickedpoint)
        self.dispload.clicked.connect(self.load_dispdata)
        self.pushButton.clicked.connect(self.plot_sen)
        # self.disp_plot = InteractiveDispPlot()
        # self.verticalLayout_3.addWidget(self.disp_plot)
        # self.progress_bar.setTextVisible(True)  # 显示默认文本
        header = self.paramshow.horizontalHeader()
        # 设置最后一列自动拉伸
        header.setStretchLastSection(True)
        header1 = self.dataproperties.horizontalHeader()
        # 设置最后一列自动拉伸
        header1.setStretchLastSection(True)
        header2 = self.paramshow_disp.horizontalHeader()
        # 设置最后一列自动拉伸
        header2.setStretchLastSection(True)
        self.comboBox.addItem('FK')
        self.comboBox.addItem('Phaseshift')
        self.comboBox.addItem('cc_fj')
        self.cut_temp_f = []
        # self.cut_temp_allright = []
        self.cut_temp = []
        self.cut_temp_fk = np.empty((0, 2), dtype=float)
        self.disp_temp = []
        self.annotations = []
        self.setup_signals()
        self._translate_ui_buttons_to_english()
        # self.save_snapshot()

    @staticmethod
    def _contains_cjk(text):
        if not isinstance(text, str) or text == "":
            return False
        return re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text) is not None

    def _translate_ui_buttons_to_english(self):
        """
        Convert Chinese button captions to English.
        Keep existing English captions unchanged.
        """
        word_map = {
            "加载": "Load",
            "保存": "Save",
            "运行": "Run",
            "停止": "Stop",
            "开始": "Start",
            "下一张": "Next",
            "上一张": "Prev",
            "上一道": "Prev",
            "下一道": "Next",
            "参数": "Params",
            "反演": "Inversion",
            "频散": "Dispersion",
            "拾取": "Pick",
            "绘制": "Plot",
            "打开": "Open",
            "关闭": "Close",
            "导入": "Import",
            "导出": "Export",
            "读取": "Read",
            "计算": "Compute",
            "切点": "Cut",
            "切割": "Cut",
            "互相关": "CC",
            "台站": "Station",
            "敏感核": "Sensitivity",
            "成像": "Imaging",
            "窗口": "Window",
            "数据": "Data",
            "曲线": "Curve",
            "图像": "Image",
            "模式": "Mode",
            "全选": "All",
        }
        phrase_map = {
            "加载参数": "Load Params",
            "保存参数": "Save Params",
            "加载数据": "Load Data",
            "保存数据": "Save Data",
            "加载频散": "Load Dispersion",
            "保存频散": "Save Dispersion",
            "开始计算": "Start",
            "停止计算": "Stop",
            "打开反演参数": "Inversion Params",
            "反演参数": "Inversion Params",
            "打开拾取窗口": "Pick Window",
            "下一道数据": "Next",
            "上一道数据": "Prev",
            "绘制敏感核": "Plot Sensitivity",
            "绘制": "Plot",
        }

        for btn in self.findChildren(QPushButton):
            old_text = btn.text()
            if not isinstance(old_text, str) or old_text.strip() == "":
                continue
            # Skip captions that are already pure English/numeric/symbols.
            if not self._contains_cjk(old_text):
                continue

            text = old_text.strip()
            new_text = phrase_map.get(text, text)
            if new_text == text:
                for cn, en in word_map.items():
                    if cn in new_text:
                        new_text = new_text.replace(cn, en)
                # Normalize multiple spaces
                new_text = re.sub(r"\s+", " ", new_text).strip()

            # If still contains CJK after replacement, keep a safe English fallback.
            if self._contains_cjk(new_text):
                # Preserve intent for known control ids.
                obj_name = btn.objectName().lower()
                if "load" in obj_name:
                    new_text = "Load"
                elif "save" in obj_name:
                    new_text = "Save"
                elif "run" in obj_name:
                    new_text = "Run"
                elif "stop" in obj_name:
                    new_text = "Stop"
                elif "pick" in obj_name:
                    new_text = "Pick"
                elif "plot" in obj_name:
                    new_text = "Plot"
                else:
                    new_text = "Action"

            btn.setText(new_text)

    def save_snapshot(self):
        self.save_highdpi(self, _snapshot_path("main_window.png"), scale=10)

    def save_highdpi(self, widget, filename, scale=4.0):
        size = widget.size()

        pixmap = QPixmap(size * scale)
        pixmap.setDevicePixelRatio(scale)

        painter = QPainter(pixmap)
        painter.setRenderHints(
            QPainter.Antialiasing |
            QPainter.TextAntialiasing |
            QPainter.SmoothPixmapTransform
        )

        widget.render(painter)
        painter.end()

        pixmap.save(filename)
        print("Saved:", filename)

    def _extract_selected_mode_freqs(self, pic_num, mode):
        try:
            keys = list(self.disp_temp_all.keys())
            if pic_num < 0 or pic_num >= len(keys):
                return np.array([], dtype=float)
            raw = np.asarray(self.disp_temp_all[keys[pic_num]], dtype=float)
        except Exception:
            return np.array([], dtype=float)

        if raw.ndim != 2:
            return np.array([], dtype=float)
        if raw.shape[0] == 2:
            disp = raw
        elif raw.shape[1] == 2:
            disp = raw.T
        else:
            return np.array([], dtype=float)

        try:
            mode_targets = split_dispersion_modes_2xn(disp, max_modes=None)
        except Exception:
            mode_targets = {}

        if isinstance(mode_targets, dict) and len(mode_targets) > 0:
            if int(mode) in mode_targets:
                mode_arr = mode_targets[int(mode)]
            else:
                nearest_key = sorted(mode_targets.keys())[0]
                mode_arr = mode_targets[nearest_key]
            freq = np.asarray(mode_arr[1], dtype=float)
        else:
            freq = np.asarray(disp[1], dtype=float)

        freq = freq[np.isfinite(freq) & (freq > 0)]
        if freq.size == 0:
            return np.array([], dtype=float)
        return np.unique(freq)

    def plot_sen(self):
        if self.vel_model is None:
            print("No inversion model is available for sensitivity calculation.")
            return
        try:
            # 计算图像索引；为空时默认使用当前道
            station_text = str(self.sen_pick.text()).strip()
            if station_text == "":
                pic_num = int(self.i_pic)
            else:
                pic_num = int(int(station_text) / int(self.params['step']))
        except ValueError:
            print("Invalid input. Enter a valid number.")
            return
        if pic_num < 0 or pic_num >= len(self.vel_model):
            print("The sensitivity channel is outside the available range.")
            return

        # Simplified UI: use default mode/param.
        mode = 0
        param_filter = "all"
        freqs = self._extract_selected_mode_freqs(pic_num, mode)
        periods = (1.0 / freqs) if freqs.size else None
        self.free_plot.plot_sen(
            pic_num * int(self.params['step']),
            self.vel_model[pic_num],
            mode=mode,
            param_filter=param_filter,
            periods=periods,
        )

    def LTSC(self):
        self.LTSC_window = LTSC_Window()
        self.LTSC_window.show()

    def stop_cc(self):
        self.thread.stop()

    def FKdata_update(self, data):
        self.cc_data[int(self.i_pic)] = data

    def FKdata(self, *_):
        if self.cc_data is None or len(self.cc_data) == 0:
            return
        pic_idx = int(self.i_pic)
        if pic_idx < 0 or pic_idx >= len(self.cc_data):
            return

        view_mode = self.free_plot.get_analysis_mode()
        if view_mode == "sensitivity":
            self.plot_sen()
            return

        file_params = self.cc_file_params or {}
        if file_params.get("delta") is not None:
            dt = float(file_params["delta"])
        elif file_params.get("samplerate") is not None:
            dt = 1.0 / float(file_params["samplerate"])
        elif self.params.get("delta") is not None:
            dt = float(self.params["delta"])
        elif self.params.get("samplerate") is not None:
            dt = 1.0 / float(self.params["samplerate"])
        else:
            dt = float(self.params_disp.get("dt", self.params_all['disp']['dt']))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid sampling interval for spatial analysis: {dt}")

        dx = float(file_params.get("dr", self.params.get("dr", self.params_disp.get("dr", 1.0))))
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError(f"Invalid receiver spacing for spatial analysis: {dx}")

        if view_mode == "beamforming":
            cc_panel = self.cc_data[pic_idx]
            fk_cutpoint = self._get_active_cutpoint()
            if fk_cutpoint.size:
                cc_panel = self._apply_cutpoint_to_cc_data(cc_panel, self.params, fk_cutpoint)
            self.free_plot.plot_beamforming(
                cc_panel,
                dt,
                dx,
                pic_idx,
                self.params,
                self.params_all.get('disp', {}),
            )
            return

        cc_panel = self.cc_data[pic_idx]
        fk_cutpoint = self._get_active_cutpoint()
        if fk_cutpoint.size:
            cc_panel = self._apply_cutpoint_to_cc_data(cc_panel, self.params, fk_cutpoint)
        self.free_plot.fk_filter(cc_panel, dt, dx, w=0)
        self.free_plot.plot(self.i_pic, self.params)

    def handle_output(self, text):
        """处理输出文本"""
        self.text_edit.moveCursor(self.text_edit.textCursor().End)
        self.text_edit.insertPlainText(text)
        self.text_edit.ensureCursorVisible()

    def load_cc(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "/cc_temp.h5", "All Files (*)")
            if file_path:  # 只有当用户实际选择了文件夹时才会更新参数
                self.cc_data, loaded_params = read_single_h5(file_path)
                previous_file_params = self.cc_file_params
                yaml_cc_params = dict(getattr(self, "params_all", {}).get("cc", {}))
                if yaml_cc_params:
                    self.params = yaml_cc_params
                elif previous_file_params and self.params:
                    self.params = {
                        key: value
                        for key, value in self.params.items()
                        if key not in previous_file_params
                    }
                self.cc_file_path = file_path
                self.cc_file_params = dict(loaded_params or {})
                self.params_disp_1T = self.cc_file_params
                if self.params is None:
                    self.params = {}
                self.params.update(self.cc_file_params)
                if getattr(self, "params_disp", None) is None:
                    self.params_disp = dict(getattr(self, "params_all", {}).get("disp", {}))
                self.params_disp = resolve_dispersion_params(
                    self.params_disp,
                    self.params,
                    self.cc_file_params,
                )
                self.load_params_update(self.params)
                self.load_params_to_table_disp(self.params_disp)
                applied = [key for key in CC_DATA_PARAMETER_KEYS if key in self.cc_file_params]
                if applied:
                    print("Dispersion parameters read from CCF file: " + ", ".join(applied))
                QMessageBox.information(self, "Notice", "Cross-correlation data loaded successfully.")
            else:
                return
        except Exception as exc:
            print(f"Failed to load CCF data: {exc}")
            return

    def save_ccdata(self):
        # try:
        # 打开文件夹选择对话框
        # folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "/cc_temp.h5", "All Files (*)")
        if file_name:  # 只有当用户实际选择了文件夹时才会更新参数
            write_single_h5(file_name, self.cc_data, self.params)
        else:
            return

    def savedisp_picdata(self):
        # self.folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        self.file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "/disp_temp.h5", "All Files (*)")
        if self.file_name:  # 只有当用户实际选择了文件夹时才会更新参数
            E_all, extent = self.E_allextent
            self.params_disp['extent'] = extent
            write_single_h5(self.file_name, E_all, self.params_disp)
            QMessageBox.information(self, "Notice", "Dispersion data saved successfully.")
        else:
            return

    def savedisp_pickedpoint(self):
        try:
            self.disp_temp_all['%d' % int(int(self.pic_num) * int(self.params['step']))] = np.array(self.disp_temp)
            # QMessageBox.warning(self, "提示", "Disp data saved successfully!!!")
            self.params_all['inv_para']['ML'] = False
            # write_single_h5(self.folder_path + 'pick_temp.h5', self.disp_temp_all, self.params_disp)
            if self.first_call:
                # self.folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
                self.file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "/pick_temp.pkl", "All Files (*)")
                self.first_call = False
            with open(self.file_name, "wb") as f:
                pickle.dump(self.disp_temp_all, f)

            self.messgeshow.setText(
                'Channel %d picked point is saved successfully ' % int(self.pic_num * int(self.params['step'])))
        except:
            pass

    def load_dispdata(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Pick File", "/pick_temp.pkl", "All Files (*)")
            file_path1, _ = QFileDialog.getOpenFileName(self, "Select Dispersion File", "/disp_temp.h5", "All Files (*)")
            if file_path:  # 只有当用户实际选择了文件夹时才会更新参数
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        self.disp_temp_all = pickle.load(f)
            else:
                pass
            if file_path1:  # 只有当用户实际选择了文件夹时才会更新参数
                if os.path.exists(file_path1):
                    E_all, self.params_disp = read_single_h5(file_path1)
                    self.E_allextent = [E_all, self.params_disp['extent']]

                    QMessageBox.information(self, "Notice", "Dispersion data loaded successfully.")
        except:
            return

    def save_disp2txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select picked dispersion data",
            "",
            "Pickle files (*.pkl);;All files (*)",
        )
        if not path:
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return

        with open(path, "rb") as f:
            disp_temp_all = pickle.load(f)

        for key, points in disp_temp_all.items():
            points = np.asarray(points)
            output_path = os.path.join(output_dir, f"dispersion_{key}.txt")
            np.savetxt(
                output_path,
                np.hstack((points, np.ones((len(points), 1)))),
                fmt="%.6f",
            )

    def calculate_noise_psd(self, data, fs=1.0, nperseg=256):
        """
        f : ndarray - 频率数组
        Pxx : ndarray - 对应的功率谱密度
        """
        f, Pxx = signal.welch(data,
                              fs=fs,
                              window='hann',
                              nperseg=nperseg,
                              noverlap=nperseg // 2,
                              scaling='density',
                              detrend='constant')
        return f, Pxx

    def receive_data(self, data):
        start_channel = int(np.asarray(self.params.get('range', [0]))[0])
        for i in range(len(self.E_allextent[0])):
            channel = start_channel + i * int(self.params['step'])
            self.disp_temp_all[str(channel)] = np.asarray(data[i], dtype=float)
            self.params_all['inv_para']['ML'] = True

    def inv_data(self, vel_model):

        # 在主窗口中处理传递过来的数据
        self.vel_model = vel_model

    def params_inv_ud(self, params):

        self.params_inv.update(params)
        self.params_inv = _prune_inv_params(self.params_inv)
        # 保存 YAML
        try:
            self.params_all['inv_para'] = self.params_inv
            with open(self.file_path, "w", encoding="utf-8") as f:
                yaml.dump(self.params_all, f)
            print("YAML file saved")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")
            print("Failed to save YAML:", e)

    def turn_last_cc(self):
        # try:
            self.cc_plot.current_scatter_data = None
            self.i_pic = self.i_pic - 1
            self.pic_num_cc = self.i_pic * self.params['step'] + int(self.params['b_ch'])
            self.picchose.setText(str(int(self.pic_num_cc)))  # 同步显示框
            self.plot_with_matplotlib(self.cc_data)
        # except Exception as e:
        #     QMessageBox.critical(self, '错误', f'加载文件时出错: {e}')
        #     print(f'程序出错: {e}')  # 打印错误信息到控制台

    def turn_next_cc(self):
        # try:
            self.cc_plot.current_scatter_data = None
            self.i_pic = self.i_pic + 1
            self.pic_num_cc = self.i_pic * self.params['step'] + int(self.params['b_ch'])

            self.picchose.setText(str(int(self.pic_num_cc)))  # 同步显示框
            self.plot_with_matplotlib(self.cc_data)

        # except Exception as e:
        #     QMessageBox.critical(self, '错误', f'error: {e}')
        #     print(f'程序出错: {e}')  # 打印错误信息到控制台

    def turn_last_disp(self):
        try:
            self.disp_plot.current_scatter_data = None
            self.pic_num = self.pic_num - 1
            self.picchose_disp.setText(str(int(self.pic_num * self.params['step'])))  # 同步显示框
            self.plot_disp(self.E_allextent)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')
            print(f'Application error: {e}')

    def turn_next_disp(self):
        try:
            self.disp_plot.current_scatter_data = None
            self.pic_num = self.pic_num + 1
            self.picchose_disp.setText(str(int(self.pic_num * self.params['step'])))  # 同步显示框
            self.plot_disp(self.E_allextent)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')
            print(f'Application error: {e}')

    def open_pick_Window(self):
        try:
            pick_params = resolve_dispersion_params(
                self.params_disp,
                self.params,
                self.cc_file_params,
            )
            for key in (
                    "picker_model_path",
                    "pick_threshold",
                    "pick_min_area",
                    "pick_min_width",
                    "pick_max_gap",
                    "pick_smooth_sigma",
                    "pick_sample_step",
            ):
                if key in self.params and key not in pick_params:
                    pick_params[key] = self.params[key]
            step = max(int(pick_params.get("step", 1)), 1)
            if "range" not in pick_params:
                image_count = int(np.asarray(self.E_allextent[0]).shape[0])
                start_channel = int(self.params.get("b_ch", 0))
                pick_params["range"] = [
                    start_channel,
                    start_channel + image_count * step,
                ]
            self.pick_window = pick_Window(pick_params, self.disp_temp_all, self.E_allextent,
                                           self.receive_data)  # 创建新窗口实例
            self.pick_window.show()  # 显示新窗口
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')
            print(f'Application error: {e}')

    def open_inv_Window(self):
        # try:
        self.params_all.setdefault('inv_para', {})
        self.params_all['inv_para'] = _prune_inv_params(self.params_all['inv_para'])
        self.params_all['inv_para'].setdefault('optimizer', 'Adam')
        self.params_all['inv_para'].setdefault('es', 1e-08)
        self.params_all['inv_para'].setdefault('init_model_depth', 60.0)
        self.params_all['inv_para'].setdefault('iter_max', 100.0)
        self.params_all['inv_para'].setdefault('model_num', 100)
        self.params_all['inv_para'].setdefault('retry_bad_channel', True)
        self.params_all['inv_para'].setdefault('retry_nstart_factor', 1.8)
        self.params_all['inv_para'].setdefault('retry_vsrange_expand', 0.18)
        self.params_all['inv_para'].setdefault('retry_thickness_expand', 0.12)
        self.params_all['inv_para'].setdefault('disp_resample_mode', 'logf')
        self.params_all['inv_para'].setdefault('disp_resample_points', 32)
        self.params_all['inv_para'].setdefault('inv_profile_smooth', 1)
        self.params_all['inv_para'].setdefault('device', 'cuda')
        self.params_all['inv_para']['inv_object'] = _resolve_inv_object(self.params_all['inv_para'], default="vs")
        self.params_inv = dict(self.params_all['inv_para'])
        self.params_inv.update(
            {'fmin': self.params_disp['fmin'], 'fmax': self.params_disp['fmax'], 'vmax': self.params_disp['vmax'],
             'vmin': self.params_disp['vmin'], 'step': self.params['step']})
        self.params_inv = _prune_inv_params(self.params_inv)
        self.params_all['inv_para'] = dict(self.params_inv)
        self.inv_window = inv_Window(self.params_all['inv_para'], self.disp_temp_all, self.inv_data,
                                     self.params_inv_ud)
        self.inv_window.show()  # 显示新窗口

    # except Exception as e:
    #     QMessageBox.critical(self, '错误', f'加载文件时出错: {e}')
    #     print(f'程序出错: {e}')  # 打印错误信息到控制台

    def disp_inv(self):
        # self.disp_temp
        self.target = np.hsplit(np.array(self.disp_temp), 2)
        self.INV = grad_cal(self.params_inv,
                            self.target,
                            AK135_data=[],
                            device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))

    def cut_save_func(self, data):
        try:
            arr = np.asarray(data, dtype=float)
        except Exception:
            arr = np.empty((0, 2), dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2) if arr.size % 2 == 0 else np.empty((0, 2), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            arr = np.empty((0, 2), dtype=float)
        self.cut_temp = arr

    def fk_cut_save_func(self, data):
        try:
            arr = np.asarray(data, dtype=float)
        except Exception:
            arr = np.empty((0, 2), dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2) if arr.size % 2 == 0 else np.empty((0, 2), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            arr = np.empty((0, 2), dtype=float)
        self.cut_temp_fk = arr

    def fk_cut_processed_update(self, data):
        try:
            self.FK_DATA = np.asarray(data, dtype=float)
        except Exception:
            self.FK_DATA = data

    def _build_fk_cut_mask(self, k_axis, f_axis, cut_points):
        pts = np.asarray(cut_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
            return None

        k_pts = pts[:, 0]
        f_pts = pts[:, 1]
        pos_mask = k_pts > 0
        neg_mask = k_pts < 0
        if np.count_nonzero(pos_mask) < 2 or np.count_nonzero(neg_mask) < 2:
            return None

        try:
            slope_pos, intercept_pos = np.polyfit(f_pts[pos_mask], k_pts[pos_mask], 1)
            slope_neg, intercept_neg = np.polyfit(f_pts[neg_mask], k_pts[neg_mask], 1)
        except Exception:
            return None

        k_pos = f_axis * slope_pos + intercept_pos
        k_neg = f_axis * slope_neg + intercept_neg
        k_pos = np.maximum(k_pos, 0.0)
        k_neg = np.minimum(k_neg, 0.0)

        mask = np.zeros((len(k_axis), len(f_axis)), dtype=float)
        for jf, f in enumerate(f_axis):
            left = min(k_neg[jf], k_pos[jf])
            right = max(k_neg[jf], k_pos[jf])
            keep = (k_axis >= left) & (k_axis <= right)
            mask[:, jf] = keep.astype(float)
        return mask

    def _active_cut_storage_name(self):
        try:
            mode = str(self.free_plot.get_analysis_mode()).strip().lower()
        except Exception:
            mode = "fk"
        if mode == "fk":
            return "x_cut_fk"
        return "x_cut"

    def _get_active_cutpoint(self):
        name = self._active_cut_storage_name()
        value = getattr(self, name, [])
        cutpoint = np.asarray(value, dtype=float) if has_len(value) else np.array([])
        if cutpoint.ndim == 2 and cutpoint.shape[0] == 2 and cutpoint.shape[1] > 0:
            return cutpoint
        return np.array([])

    def _apply_cutpoint_to_cc_data(self, cc_panel, params_like, cutpoint):
        data = np.asarray(cc_panel, dtype=float).copy()
        cutpoint = np.asarray(cutpoint, dtype=float)
        if data.ndim != 2:
            return data
        if cutpoint.ndim != 2 or cutpoint.shape[0] != 2 or cutpoint.shape[1] == 0:
            return data

        try:
            dt = 1.0 / float(params_like['samplerate_in_use'])
            cc_len = float(params_like['cc_len'])
            len_t = int((cc_len / dt) / 2)
        except Exception:
            return data

        ncut = cutpoint.shape[1]
        nsamp = data.shape[-1]
        for i_sta in range(data.shape[0]):
            amp = float(np.max(np.abs(data[i_sta])))
            if amp > 0:
                data[i_sta] /= amp

            cut_idx = min(i_sta, ncut - 1)
            t_neg = float(cutpoint[0, cut_idx])
            t_pos = float(cutpoint[1, cut_idx])
            if np.isfinite(t_neg) and np.isfinite(t_pos):
                right_start = min(int(len_t + t_pos / dt), nsamp)
                left_end = max(int(len_t + t_neg / dt), 0)
                data[i_sta, right_start:] = 0
                data[i_sta, :left_end] = 0

            c0 = max(int(len_t) - 10, 0)
            c1 = min(int(len_t) + 10, nsamp)
            data[i_sta, c0:c1] = 0
        return data

    def start_run_disp(self):
        try:
            current_text = self.comboBox.currentText()
            params = resolve_dispersion_params(
                self.params_disp,
                self.params,
                self.cc_file_params,
            )
            use_cut = True
            try:
                use_cut = bool(self.disp_plot.use_cut_enabled())
            except Exception:
                use_cut = True
            if use_cut:
                cutpoint = np.asarray(self.x_cut, dtype=float) if has_len(self.x_cut) else np.array([])
                if cutpoint.ndim == 2 and cutpoint.shape[0] == 2 and cutpoint.shape[1] > 0:
                    params['cutpoint'] = cutpoint
                else:
                    params['cutpoint'] = []
            else:
                params['cutpoint'] = []
            params['method'] = current_text
            self.params_disp = dict(params)
            start_sta = int(np.ceil(params['range'][0] / params['step']))
            end_sta = int(np.floor(params['range'][1] / params['step']))
            source_count = np.asarray(self.cc_data).squeeze().shape[0]
            requested_count = max(end_sta - start_sta, 0)
            available_count = max(source_count - max(start_sta, 0), 0)
            self.dispbar.setMaximum(min(requested_count, available_count) or source_count)
            self.thread_disp = run_disp(params, self.cc_data)
            self.thread_disp.progress_signal.connect(self.update_disp_progress)
            self.thread_disp.progress_data.connect(self.handle_disp_progress_data)
            self.thread_disp.progress_str1.connect(self.show_ccmessage)
            # self.thread_disp.task_finished.connect(self.on_task_finished)
            self.thread_disp.finished_signal.connect(self.show_ccmessage)
            self.thread_disp.start()

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file: {e}')
            print(f'Application error: {e}')

    # def on_task_finished(self):
    #     self.plot_disp(self.E_allextent)

    def start_run_cc(self):
        # try:
        #     # self.params = self.params_all['cc']
        # except Exception as e:
        #     QMessageBox.critical(self, '错误', f'加载文件时出错: {e}')
        #     print(f'程序出错: {e}')  # 打印错误信息到控制台
        #     return
        # data, params_temp = read_data(params['example_path'])  # 数据和参数读取
        # params.update(params_temp)  # 参数更新
        if str(self.params.get('datatype', '')).lower() == 'h5':
            with h5py.File(self.params['example_path'], 'r') as f:
                data_struct = finddatastruct(f)
            self.params_temp = read_data(self.params['example_path'], data_struct=data_struct, para=True,
                                         data_temp=False)  # 数据和参数读取
            self.params.update(self.params_temp)  # 参数更新
            self.params_temp.update(self.params)

        # params.update(self.params_temp)  # 参数更新
        self.load_properties()
        self.ccbar.setRange(0, 100)
        self.ccbar.setValue(0)

        # 创建线程实例，并传入数组
        self.thread = run_cc(self.params)
        # 连接进度信号到更新进度条的槽函数
        self.thread.progress_signal.connect(self.update_progress)
        # self.thread.progress_data.connect(self.plot_with_matplotlib)
        # 连接线程结束信号到槽函数
        self.thread.progress_data.connect(self.handle_progress_data)
        self.thread.finished_signal.connect(self.thread_finished)
        # 启动线程
        self.thread.progress_str1.connect(self.show_ccmessage)
        self.thread.finished_signal.connect(self.show_ccmessage)
        self.thread.progress_str2.connect(self.load_properties)
        self.thread.progress_str3.connect(self.load_properties2)
        # self.messgeshow.setText()

        self.thread.start()

    def line_to_inter(self):
        pts = np.asarray(self.cut_temp, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
            QMessageBox.warning(self, "Notice", "At least four CUT points are required, with two on each side of zero time.")
            return

        x = pts[:, 0]
        y = pts[:, 1]
        pos_mask = x > 0
        neg_mask = x < 0
        if np.count_nonzero(pos_mask) < 2 or np.count_nonzero(neg_mask) < 2:
            QMessageBox.warning(self, "Notice", "CUT points must include at least two points in both the positive and negative time windows.")
            return

        x_pos = x[pos_mask]
        y_pos = y[pos_mask]
        x_neg = x[neg_mask]
        y_neg = y[neg_mask]

        c_range = int(self.params['c_range'])
        dr = float(self.params['dr'])
        time_range = float(self.params['time_range'])
        y_new = np.arange(c_range + 1, dtype=float) * dr

        try:
            slope_pos, intercept_pos = np.polyfit(y_pos, x_pos, 1)
            slope_neg, intercept_neg = np.polyfit(y_neg, x_neg, 1)
        except Exception as e:
            QMessageBox.warning(self, "Notice", f"CUT fitting failed: {e}")
            return

        x_pos_new = np.clip(y_new * slope_pos + intercept_pos, 0.0, time_range)
        x_neg_new = np.clip(y_new * slope_neg + intercept_neg, -time_range, 0.0)

        x_cut_temp = np.vstack((x_neg_new, x_pos_new))
        x_plot = np.hstack((x_neg_new, x_pos_new))
        y_plot = np.hstack((y_new, y_new))
        self.cc_plot.current_scatter_data = np.column_stack((x_plot, y_plot))
        self.cut_temp_f = self.cc_plot.current_scatter_data
        storage_name = self._active_cut_storage_name()
        setattr(self, storage_name, x_cut_temp)
        if storage_name == "x_cut":
            self.params['cutpoint'] = self.x_cut
            plot_params = self.params
        else:
            plot_params = dict(self.params)
            plot_params['cutpoint'] = x_cut_temp

        self.cc_plot.plot(self.cc_data, int(self.i_pic), plot_params)
        # scene_pos_y = (self.params['c_range'] * self.params['dr'] - y_PLOT) / (
        #         self.params['c_range'] * self.params['dr']) * 505 + 95
        # PLOT = np.vstack((scene_pos_x, scene_pos_y)).T
        # self.i_pic
        # self.x_cut = x_cut_temp
        # for i in range(len(PLOT.T) - 1):
        #     start_point = QPointF(PLOT.T[i][0], PLOT.T[i][1])
        #     end_point = QPointF((PLOT.T[i + 1][0]), PLOT.T[i+1][1])
        #     line = QGraphicsLineItem(QLineF(start_point, end_point))
        #     line.setPen(QPen(Qt.blue, 2))  # 设置线的颜色和宽度
        #     self.scene.addItem(line)
        # self.cut_temp = []

    def show_ccmessage(self, text):
        self.messgeshow.setText(text)

    def handle_progress_data(self, data):
        """
        处理 progress_data 信号的槽函数。

        参数:
            data: 从 progress_data 信号传递的数据。
        """
        # print("Received data from thread:", data)
        self.cc_data = data
        # self.plot_with_matplotlib(cc_data=data)

    def handle_disp_progress_data(self, data):
        """
        处理 progress_data 信号的槽函数。

        参数:
            data: 从 progress_data 信号传递的数据。
        """
        # print("Received data from thread:", data)
        self.E_allextent = data
        # self.plot_with_matplotlib(cc_data=data)

    def update_progress(self, value):
        # 更新进度条的值
        self.ccbar.setValue(max(0, min(100, int(value))))

    def update_disp_progress(self, value):
        # 更新进度条的值
        start_sta = int(np.ceil(self.params_disp['range'][0] / self.params_disp['step']))
        self.dispbar.setValue(value - start_sta + 1)

    def thread_finished(self):
        print("Thread finished!")
        self.ccrun.setEnabled(True)  # 重新启用按钮

    def plot_disp(self, E_allextent):
        try:
            # self.disp_scene.clear()
            self.params_disp['range'] = np.array(list(self.params_disp['range']))
            start_sta = int(np.ceil(self.params_disp['range'][0] / self.params_disp['step']))
            end_sta = int(np.floor(self.params_disp['range'][1] / self.params_disp['step']))
            self.E_all, self.extent = E_allextent

            # 如果用户没有选择特定图像
            if not self.picchose_disp.text().strip():
                self.img_sum = []
                for i in range(len(self.E_all)):
                    E = self.E_all[i]
                    img_data = self.generate_image(E, self.extent)
                    self.img_sum.append(img_data)
            else:
                # 如果用户选择了特定图像
                try:
                    # 计算图像索引
                    pic_num = int(int(self.picchose_disp.text()) / int(self.params['step']))
                except ValueError:
                    print("Invalid input. Enter a valid number.")
                    return
                self.pic_num = pic_num
                self.disp_temp = []
                self.annotations = []
                # 生成并显示图像
                if not (start_sta <= self.pic_num < end_sta):
                    QMessageBox.critical(self, 'Error', 'Choose a number within the available range.')
                    return
                self.generate_and_display_single_image(self.pic_num - start_sta)

        except Exception as e:

            QMessageBox.critical(self, "Error", f"Failed to load parameters: {e}")

    def generate_image(self, E, extent):
        # 创建 Matplotlib 图像
        fig = plt.figure("Image", frameon=False)
        canvas = fig.canvas
        plt.imshow(E, extent=extent, aspect='auto', origin='lower', cmap='jet')
        fig.set_size_inches(256 / 100, 256 / 100)
        plt.axis('off')  # 去掉坐标轴
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 将图像保存到内存缓冲区
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        data = buffer.getvalue()

        # 将图像数据转换为 NumPy 数组
        img = Image.open(io.BytesIO(data))
        img = img.convert('RGB')
        img = np.array(img)
        return img.transpose(2, 0, 1)

    def disp_update(self, data):
        self.disp_temp = data

    def generate_and_display_single_image(self, pic_num):
        # 创建 Matplotlib 图像
        self.disp_plot.plot(
            E_all=self.E_all,
            disp_temp_all=self.disp_temp_all,
            pic_num=pic_num,
            params=self.params_disp,
            extent=self.extent
        )

    def plot_with_matplotlib(self, cc_data):
        # try:
        pic_num = int(self.picchose.text())
        step = self.params['step']
        b_ch = int(self.params['b_ch'])
        ncc = len(self.cc_data)
        i_pic = (pic_num - b_ch) // step
        i_pic = max(0, min(i_pic, ncc - 1))
        self.i_pic = i_pic
        # temp = 0
        #
        # if hasattr(self, 'i_pic_temp') and self.i_pic_temp != int(self.picchose.text()):
        #     temp = 1
        # pic_num = int(self.picchose.text())
        # # self.scene.clear()
        # i_pic = (pic_num)/ self.params['step']
        # self.i_pic_temp = pic_num
        #
        # if not hasattr(self, 'i_pic') or temp == 1:
        #     self.i_pic = i_pic-int(self.params['b_ch']/self.params['step'])
        #     self.pic_num_cc = pic_num
        #
        # if self.i_pic < 0:
        #     self.i_pic = len(self.cc_data) + self.i_pic
        # elif self.i_pic > len(self.cc_data):
        #     self.i_pic = self.i_pic - len(self.cc_data)
        # 检查 cc_data 的形状
        if cc_data.squeeze().ndim != 3:
            raise ValueError("cc_data must be a three-dimensional array")
        cc_data = cc_data.squeeze()
        self.cc_plot.plot(cc_data, int(self.i_pic), self.params)

    # except:
    #     return

    # except Exception as e:
    #     QMessageBox.critical(self, '错误', f'加载文件时出错: {e}')
    #     print(f'程序出错: {e}')  # 打印错误信息到控制台

    def select_file(self):
        """选择文件并显示文件路径"""
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Select File', '',
                                                        'YAML Files (*.yaml *.yml);;All Files (*)')
        if self.file_path:
            self.parampath.setText(self.file_path)  # 将选择的文件路径显示在输入框中

    def setup_signals(self):
        self.paramshow.itemChanged.connect(self.on_item_changed)

    def load_params_to_table(self, params: dict):
        # 暂时断开信号
        try:
            self.paramshow.itemChanged.disconnect(self.on_item_changed)
        except:
            pass

        self.paramshow.clear()
        self.paramshow.setRowCount(len(params))
        self.paramshow.setColumnCount(2)
        self.paramshow.setHorizontalHeaderLabels(["Key", "Value"])

        for row, (k, v) in enumerate(params.items()):
            key_item = QTableWidgetItem(str(k))
            val_item = QTableWidgetItem(str(v))
            self.paramshow.setItem(row, 0, key_item)
            self.paramshow.setItem(row, 1, val_item)

        # 恢复信号
        self.paramshow.itemChanged.connect(self.on_item_changed)

    def load_params_to_table_disp(self, params: dict):
        # 暂时断开信号
        try:
            self.paramshow_disp.itemChanged.disconnect(self.on_item_changed_disp)
        except:
            pass

        self.paramshow_disp.clear()
        self.paramshow_disp.setRowCount(len(params))
        self.paramshow_disp.setColumnCount(2)
        self.paramshow_disp.setHorizontalHeaderLabels(["Key", "Value"])

        for row, (k, v) in enumerate(params.items()):
            key_item = QTableWidgetItem(str(k))
            val_item = QTableWidgetItem(str(v))
            self.paramshow_disp.setItem(row, 0, key_item)
            self.paramshow_disp.setItem(row, 1, val_item)

        # 恢复信号
        self.paramshow_disp.itemChanged.connect(self.on_item_changed_disp)

    def load_params(self):
        file_path = self.parampath.text()
        # self.save_snapshot()
        if not file_path:
            QMessageBox.warning(self, "Error", "Select a YAML file first.")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    data = yaml.load(f)  # 兼容旧版本

            self.params_all = data
            self.params = self.params_all.get("cc", {}).copy()
            if self.cc_file_params:
                self.params.update(self.cc_file_params)
            self.params_inv = _prune_inv_params(self.params_all.get("inv_para", {}))
            self.params_all["inv_para"] = self.params_inv

            self.load_params_to_table(self.params)
            self.file_path = file_path

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {e}")
            print("Failed to load YAML:", e)

    def load_params_update(self, params: dict):
        self.params = params
        self.load_params_to_table(self.params)

    def on_item_changed(self, item):
        if item.column() != 1:
            return  # 只处理 Value 列
        self.paramshow.blockSignals(True)

        self.save_data()

        # 🔥 保存结束后再重新打开 signal
        self.paramshow.blockSignals(False)

    def save_data(self):

        if self.params is None:
            QMessageBox.warning(self, "Error", "No parameters have been loaded.")
            return

        original_params = self.params_all.get("cc", {})
        updated_params = {}

        for row in range(self.paramshow.rowCount()):
            key_item = self.paramshow.item(row, 0)
            val_item = self.paramshow.item(row, 1)

            if not key_item or not val_item:
                continue

            key = key_item.text().strip()
            value_str = val_item.text().strip()
            original_value = original_params.get(key)

            # 传入 val_item 以便标红
            if original_value is not None:
                updated_value = self.convert_value(value_str, original_value, val_item)
            else:
                updated_value = self._infer_type(value_str, val_item)

            updated_params[key] = updated_value

        # 更新 cc
        self.params = updated_params
        self.params_all["cc"] = updated_params

        print("\n======= Parameters Updated =======")
        for k, v in updated_params.items():
            print(f"{k}: {v}")
        print("=======================\n")

        # 保存 YAML
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                yaml.dump(self.params_all, f)
            print("YAML file saved")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")
            print("Failed to save YAML:", e)

    def _infer_type(self, s, cell_item=None):
        s = s.strip()

        # bool
        if s.lower() in ("true", "false"):
            self._mark_cell_ok(cell_item)
            return s.lower() == "true"

        # list / tuple / dict / number using literal_eval
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, dict, int, float)):
                self._mark_cell_ok(cell_item)
                return val
        except:
            pass

        # numpy array 格式 np.array([1,2,3])
        if s.startswith("np.array"):
            try:
                inner = s[s.find("(") + 1: s.rfind(")")]
                arr = np.array(ast.literal_eval(inner))
                self._mark_cell_ok(cell_item)
                return arr
            except:
                pass

        # string fallback
        self._mark_cell_ok(cell_item)
        return s

    def load_params_disp(self):
        self.params_disp = resolve_dispersion_params(
            self.params_all.get('disp', {}),
            self.params,
            self.cc_file_params,
        )

        try:
            self.load_params_to_table_disp(self.params_disp)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load parameters: {e}')
            print(f'Application error: {e}')

    def on_item_changed_disp(self, item):
        if item.column() != 1:
            return  # 只处理 Value 列
        # 🔥 关闭 signal，避免保存多次
        self.paramshow_disp.blockSignals(True)

        self.save_data_disp()

        # 🔥 保存结束后再重新打开 signal
        self.paramshow_disp.blockSignals(False)

    def save_data_disp(self):

        if self.params_disp is None:
            QMessageBox.warning(self, "Error", "No parameters have been loaded.")
            return
        original_params = self.params_all.get("disp", {})
        updated_params = {}

        for row in range(self.paramshow_disp.rowCount()):
            key_item = self.paramshow_disp.item(row, 0)
            val_item = self.paramshow_disp.item(row, 1)

            if not key_item or not val_item:
                continue

            key = key_item.text().strip()
            value_str = val_item.text().strip()
            original_value = original_params.get(key)

            # 传入 val_item 以便标红
            if original_value is not None:
                updated_value = self.convert_value(value_str, original_value, val_item)
            else:
                updated_value = self._infer_type(value_str, val_item)

            updated_params[key] = updated_value

        # 更新 cc
        self.params_disp = updated_params
        self.params_all["disp"] = updated_params

        print("\n======= Parameters Updated =======")
        for k, v in updated_params.items():
            print(f"{k}: {v}")
        print("=======================\n")
        data_clean = sanitize_for_yaml(self.params_all)
        # 保存 YAML
        # try:
        with open(self.file_path, "w", encoding="utf-8") as f:
            yaml.dump(data_clean, f)
        print("YAML file saved")
        # except Exception as e:
        #     QMessageBox.critical(self, "错误", f"保存失败: {e}")
        #     print("保存 YAML 出错:", e)

    def load_properties(self, endtime=False):
        try:
            self.params.update(self.params_temp)
            if endtime:
                self.params_temp['endtime'] = endtime
            # 设置 QTableWidget 的行数和列数
            self.dataproperties.setRowCount(len(self.params_temp))
            self.dataproperties.setColumnCount(2)
            self.dataproperties.setHorizontalHeaderLabels(['Key', 'Value'])

            # 填充 QTableWidget
            for row, (key, value) in enumerate(self.params_temp.items()):
                key_item = QTableWidgetItem(key)  # 第一列显示键
                value_item = QTableWidgetItem(str(value))  # 第二列显示值
                self.dataproperties.setItem(row, 0, key_item)
                self.dataproperties.setItem(row, 1, value_item)


        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load data properties: {e}')
            print(f'Application error: {e}')

    def load_properties2(self, starttime=False):
        try:
            if starttime:
                self.params_temp['starttime'] = starttime
            # 设置 QTableWidget 的行数和列数
            self.dataproperties.setRowCount(len(self.params_temp))
            self.dataproperties.setColumnCount(2)
            self.dataproperties.setHorizontalHeaderLabels(['Key', 'Value'])

            # 填充 QTableWidget
            for row, (key, value) in enumerate(self.params_temp.items()):
                key_item = QTableWidgetItem(key)  # 第一列显示键
                value_item = QTableWidgetItem(str(value))  # 第二列显示值
                self.dataproperties.setItem(row, 0, key_item)
                self.dataproperties.setItem(row, 1, value_item)


        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load data properties: {e}')
            print(f'Application error: {e}')

    def convert_value(self, value_str, original_value, cell_item=None):
        """根据原类型转换，同时标红错误"""
        s = value_str.strip()

        if isinstance(original_value, bool):
            v = s.lower()
            if v in ("true", "1", "yes"):
                self._mark_cell_ok(cell_item)
                return True
            elif v in ("false", "0", "no"):
                self._mark_cell_ok(cell_item)
                return False

            self._mark_cell_error(cell_item)
            return original_value
        if isinstance(original_value, int):
            try:
                val = int(s)
                self._mark_cell_ok(cell_item)
                return val
            except:
                self._mark_cell_error(cell_item)
                return original_value

        if isinstance(original_value, float):
            try:
                val = float(s)
                self._mark_cell_ok(cell_item)
                return val
            except:
                self._mark_cell_error(cell_item)
                return original_value
        if isinstance(original_value, list):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    self._mark_cell_ok(cell_item)
                    return val
            except:
                pass
            self._mark_cell_error(cell_item)
            return original_value

        if isinstance(original_value, tuple):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, tuple):
                    self._mark_cell_ok(cell_item)
                    return val
            except:
                pass
            self._mark_cell_error(cell_item)
            return original_value

        if isinstance(original_value, np.ndarray):
            try:
                val = ast.literal_eval(s)  # 解析为 list
                arr = np.array(val)
                if isinstance(arr, np.ndarray):
                    self._mark_cell_ok(cell_item)
                    return arr
            except:
                pass

            self._mark_cell_error(cell_item)
            return original_value

        self._mark_cell_ok(cell_item)
        return s

    def _mark_cell_error(self, item):
        if item is not None:
            item.setBackground(QColor(255, 100, 100))  # 红色提示

    def _mark_cell_ok(self, item):
        if item is not None:
            item.setBackground(QColor(255, 255, 255))  # 恢复白色

    def keyPressEvent(self, event):
        # 检测按下的键
        if event.key() == Qt.Key_Escape:  # 按下 'Esc' 键
            print("Application closed")
            QApplication.quit()
        # elif event.key() == Qt.Key_Enter:
        #     self.save_data()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
