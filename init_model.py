import pickle, numpy as np

# 修改为你的数据路径（保持和你当前环境一致）
from scipy.interpolate import interp1d

def convert_model_to_layer_definitions(model, density_low=2.0, density_high=2.5):
    """
    model: 20x4 数组，每行 [Vs_min, Vs_max, H_min, H_max]
    返回 layer_definitions 列表
    """
    layer_definitions = []

    for row in model:
        vs_min, vs_max, h_min, h_max = row

        layer = {
            'd': (float(h_min), float(h_max)),
            'vs': (float(vs_min), float(vs_max)),
            'density': (density_low, density_high)
        }
        layer_definitions.append(layer)

    return layer_definitions


def make_thickness(
    num_layers,
    max_depth,
    mode="piecewise_exp",
    r1=1.15,
    r2=1.40,
    ratio=0.5,
    exp_r=1.25,
    integer_output=False
):
    """
    生成厚度数组，可选择分段指数、单指数，并可选整数厚度。
    """

    # -----------------------------
    # 1. 生成指数权重
    # -----------------------------
    if mode == "piecewise_exp":
        n1 = int(num_layers * ratio)
        n2 = num_layers - n1

        w1 = r1 ** np.arange(n1)
        w2 = r2 ** np.arange(n2)
        weights = np.concatenate([w1, w2])

    elif mode == "exp":
        weights = exp_r ** np.arange(num_layers)

    else:
        raise ValueError("mode must be 'piecewise_exp' or 'exp'.")

    # -----------------------------
    # 2. 归一化 → 总厚度 = max_depth
    # -----------------------------
    thickness = max_depth * weights / weights.sum()

    # -----------------------------
    # 3. 整数化处理（可选）
    # -----------------------------
    if integer_output:
        thk_int = np.round(thickness).astype(int)

        diff = max_depth - thk_int.sum()

        if diff != 0:
            step = 1 if diff > 0 else -1
            diff_abs = abs(diff)
            for i in range(diff_abs):
                # 在深层调整（不破坏浅层分辨率）
                thk_int[-1 - (i % num_layers)] += step

        thickness = thk_int.astype(float)

    return thickness


def make_forced_model(max_depth=80.0, num_layers=8):
    """
    强制生成一个适合你那张图的搜索模型（忽略data）
    Vs范围参考红线：浅层~200-300 → 中间逐步跳到~1200-1300
    """
    # 厚度均匀分配
    thickness = np.full(num_layers, max_depth / num_layers, dtype=float)  # m
    H = thickness / 1000.0  # km

    # ─────────────── 核心：手动设定参考Vs（中值） ───────────────
    # 尽量贴近红线的主要台阶
    ref_vs = np.array([
        240, 240, 240, 240, 240,          # 0~20m 很低速
        420, 420, 550, 550,               # ~20~40m 过渡
        720, 720, 950, 950, 950,          # ~40~60m 中高速
        1180, 1180, 1250, 1300, 1320      # ~60~80m+ 接近基底
    ])

    # 保证长度匹配
    if len(ref_vs) != num_layers:
        ref_vs = np.interp(
            np.linspace(0, num_layers-1, num_layers),
            np.linspace(0, len(ref_vs)-1, len(ref_vs)),
            ref_vs
        )

    # ─────────────── 生成搜索范围 ───────────────
    # 浅层低速强制（波导特征）
    n_low = max(4, int(0.25 * num_layers))       # 前25% 或至少4层很低
    Vs_min = ref_vs * 0.70
    Vs_max = ref_vs * 1.55

    # 强制浅层特别低
    Vs_min[:n_low] = np.minimum(Vs_min[:n_low], 180)
    Vs_max[:n_low] = np.minimum(Vs_max[:n_low], 380)

    # 低速层下面明显跳升（至少到1.3~1.6倍）
    jump_idx = n_low
    Vs_min[jump_idx] = max(Vs_min[jump_idx], 380)
    Vs_max[jump_idx] = max(Vs_max[jump_idx], 680)

    # 深部强制非递减（单调递增搜索空间）
    for i in range(jump_idx + 1, num_layers):
        Vs_min[i] = max(Vs_min[i], Vs_min[i-1])
        Vs_max[i] = max(Vs_max[i], max(Vs_max[i-1], Vs_min[i] + 40))

    # 最小下限保护 & 范围不要太离谱
    Vs_min = np.maximum(Vs_min, 40)
    Vs_max = np.maximum(Vs_max, Vs_min + 120)

    # 堆叠成 Dinver/ Neighborhood 常用格式
    # [Vs_min, Vs_max, thickness(m→km), thickness(m→km)]
    model = np.vstack([Vs_min, Vs_max, H, H]).T.astype(float)

    return model, thickness


def make_models(
    data,
    max_depth,
    num_layers=10,
    vs_step=50,
    alpha=0.4,
    is_period=False,
):

    models = {}
    valid_curves = {}
    for raw_key, raw_arr in data.items():
        try:
            key_num = float(raw_key)
        except Exception:
            continue
        arr0 = np.asarray(raw_arr, dtype=float)
        if arr0.ndim != 2 or arr0.shape[1] < 2 or arr0.shape[0] < 2:
            continue
        arr0 = arr0[:, :2]
        mask0 = np.isfinite(arr0[:, 0]) & np.isfinite(arr0[:, 1])
        arr0 = arr0[mask0]
        if arr0.shape[0] < 2:
            continue
        arr0 = arr0[np.argsort(arr0[:, 0])]
        _, uniq_idx = np.unique(arr0[:, 0], return_index=True)
        arr0 = arr0[np.sort(uniq_idx)]
        if arr0.shape[0] < 2:
            continue
        valid_curves[key_num] = arr0
    valid_keys_sorted = np.array(sorted(valid_curves.keys()), dtype=float)

    def _neighbor_curve(raw_key):
        try:
            k = float(raw_key)
        except Exception:
            return None
        if valid_keys_sorted.size == 0:
            return None
        right_idx = int(np.searchsorted(valid_keys_sorted, k, side='left'))
        left_idx = right_idx - 1
        left_key = valid_keys_sorted[left_idx] if left_idx >= 0 else None
        right_key = valid_keys_sorted[right_idx] if right_idx < valid_keys_sorted.size else None

        if left_key is None and right_key is None:
            return None
        if left_key is None:
            return valid_curves[float(right_key)].copy()
        if right_key is None:
            return valid_curves[float(left_key)].copy()
        if float(left_key) == float(right_key):
            return valid_curves[float(left_key)].copy()

        left_curve = valid_curves[float(left_key)]
        right_curve = valid_curves[float(right_key)]
        x_lo = max(float(np.min(left_curve[:, 0])), float(np.min(right_curve[:, 0])))
        x_hi = min(float(np.max(left_curve[:, 0])), float(np.max(right_curve[:, 0])))
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
            if abs(k - float(left_key)) <= abs(float(right_key) - k):
                return left_curve.copy()
            return right_curve.copy()

        n_out = int(max(2, min(left_curve.shape[0], right_curve.shape[0], 32)))
        x_new = np.linspace(x_lo, x_hi, n_out)
        y_left = np.interp(x_new, left_curve[:, 0], left_curve[:, 1])
        y_right = np.interp(x_new, right_curve[:, 0], right_curve[:, 1])
        w = (k - float(left_key)) / (float(right_key) - float(left_key))
        w = float(np.clip(w, 0.0, 1.0))
        y_new = (1.0 - w) * y_left + w * y_right
        return np.column_stack((x_new, y_new))

    for key, arr in data.items():

        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            arr = np.empty((0, 2), dtype=float)
        else:
            arr = arr[:, :2]
            arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]

        if arr.shape[0] < 2:
            neighbor_arr = _neighbor_curve(key)
            if neighbor_arr is not None and neighbor_arr.shape[0] >= 2:
                arr = neighbor_arr
            else:
                c0 = max(float(vs_step) * 4.0, 200.0)
                arr = np.array([[1.0, c0], [2.0, c0 * 1.1]], dtype=float)

        x = arr[:, 0]
        c = arr[:, 1]

        # =========================
        # 1️⃣ 频率处理
        # =========================
        if is_period:
            f = 1.0 / x
        else:
            f = x

        mask = np.isfinite(f) & np.isfinite(c) & (f > 0)
        f = f[mask]
        c = c[mask]

        # =========================
        # 2️⃣ 深度映射
        # =========================
        depth = alpha * c / f

        mask = np.isfinite(depth) & (depth >= 0) & (depth <= max_depth)
        depth = depth[mask]
        c = c[mask]
        if depth.size == 0 or c.size == 0:
            c_valid = np.asarray(arr[:, 1], dtype=float)
            c_valid = c_valid[np.isfinite(c_valid)]
            c0 = float(np.nanmedian(c_valid)) if c_valid.size else max(float(vs_step) * 4.0, 200.0)
            depth = np.array([0.0, float(max_depth)], dtype=float)
            c = np.array([c0 * 0.95, c0 * 1.05], dtype=float)

        # 排序
        idx = np.argsort(depth)
        depth = depth[idx]
        c = c[idx]
        if depth.size == 1:
            depth = np.array([0.0, float(max_depth)], dtype=float)
            c = np.array([float(c[0]), float(c[0])], dtype=float)

        # =========================
        # 3️⃣ 层深
        # =========================
        layer_depth = np.linspace(0, max_depth, num_layers)

        # =========================
        # 4️⃣ 插值
        # =========================
        Vs = np.interp(layer_depth, depth, c)

        # =========================
        # 5️⃣ 强平滑（关键）
        # =========================
        # 移动平均
        window = 3
        Vs_smooth = np.convolve(Vs, np.ones(window)/window, mode='same')

        # =========================
        # 6️⃣ 单调化（弱约束）
        # =========================
        depth_ratio = layer_depth / max(max_depth, 1.0)
        deep_ref = max(float(np.max(c)), 600.0)
        deep_floor = np.interp(
            depth_ratio,
            [0.0, 0.35, 0.65, 1.0],
            [float(np.min(c)) * 0.95, float(np.min(c)) * 1.00, deep_ref * 0.90, deep_ref * 1.05],
        )
        Vs_smooth = np.maximum(Vs_smooth, deep_floor)

        for i in range(1, len(Vs_smooth)):
            if Vs_smooth[i] < Vs_smooth[i-1] * 0.9:
                Vs_smooth[i] = Vs_smooth[i-1] * 0.9

        # =========================
        # 7️⃣ 转换为 min/max（收紧范围）
        # =========================
        Vs_min = np.round(Vs_smooth * 0.5 / vs_step) * vs_step
        Vs_max = np.round(Vs_smooth * 1.9 / vs_step) * vs_step

        deep_min_floor = np.interp(
            depth_ratio,
            [0.0, 0.35, 0.65, 1.0],
            [vs_step, 300.0, 550.0, 700.0],
        )
        Vs_min = np.maximum(Vs_min, np.round(deep_min_floor / vs_step) * vs_step)
        Vs_min = np.maximum(Vs_min, vs_step)
        Vs_max = np.maximum(Vs_max, Vs_min + 2 * vs_step)

        # =========================
        # 8️⃣ 厚度
        # =========================
        thickness = np.full(num_layers, max_depth / num_layers)
        H = thickness / 1000.0

        models[key] = np.vstack([Vs_min, Vs_max, H, H]).T.astype(float)

    return models


def make_models1(
    data,
    max_depth,
    num_layers=10,
    step_layers=1,
    vs_step=50,
):

    models = {}

    for key, arr in data.items():
        c = np.asarray(arr)[:, 1]

        shallow_c = float(np.min(c))
        deep_c = float(np.max(c))

        # === 初始 Vs ===
        Vs_surface = max(40, round((shallow_c * 0.95) / vs_step) * vs_step)
        Vs_bottom  = max(Vs_surface + vs_step, round((deep_c * 1.10) / vs_step) * vs_step)

        # === 阶梯数 ===
        n_steps = max(1, num_layers // step_layers)

        # === 阶梯 Vs ===
        step_vs = np.linspace(Vs_surface, Vs_bottom, n_steps)
        target_vs = np.repeat(step_vs, step_layers)[:num_layers]

        # === Vs_min / Vs_max（先不单调）===
        Vs_min = np.round(target_vs * 0.68 / vs_step) *vs_step
        Vs_max = np.round(target_vs * 2.00 / vs_step) * vs_step

        Vs_min = np.maximum(Vs_min, vs_step)
        Vs_max = np.maximum(Vs_max, Vs_min + 2 * vs_step)

        # -------------------------------------------------
        # ★ 关键修改 1：强制浅层低速层（波导）
        # -------------------------------------------------
        n_low = max(2, int(0.25 * num_layers))  # 至少 2 层低速层

        Vs_min[:n_low] = Vs_surface * 0.8
        Vs_max[:n_low] = Vs_surface * 1.0

        # -------------------------------------------------
        # ★ 关键修改 2：低速层下方必须明显变快
        # -------------------------------------------------
        Vs_min[n_low] = max(Vs_min[n_low], Vs_surface * 1.1)
        Vs_max[n_low] = max(Vs_max[n_low], Vs_surface * 1.2)

        # -------------------------------------------------
        # ★ 关键修改 3：只在深部强制单调
        # -------------------------------------------------
        for i in range(n_low + 1, num_layers):
            Vs_min[i] = max(Vs_min[i], Vs_min[i-1])
            Vs_max[i] = max(Vs_max[i], Vs_max[i-1])

        # === 厚度：均分（你已经改对了）===
        thickness = np.full(num_layers, max_depth / num_layers, dtype=float)

        H = thickness / 1000.0

        models[key] = np.vstack([Vs_min, Vs_max, H, H]).T.astype(float)

    return models

# def make_models(data, max_depth, num_layers=16, step_layers=1, vs_step=40,    thk_mode="piecewise_exp",      # "piecewise_exp" or "exp"
#     integer_thk=False,             # True → 整数厚度
#     r1=1.15,
#     r2=1.40,
#     ratio=0.4,
#     exp_r=1.25):
#     """
#     生成阶梯状模型，可调层数。
#     - num_layers: 总层数（如 6, 8, 10, 12 ...）
#     - step_layers: 每个阶梯的层数（如 2 表示每2层一个台阶）
#     - vs_step: 每个阶梯 Vs 增加的量
#     """
#
#     models = {}
#
#     for key, arr in data.items():
#         c = np.asarray(arr)[:, 1]
#
#         shallow_c = float(np.min(c))
#         deep_c = float(np.max(c))
#
#         # === 初始 Vs，限制为50整数倍 ===
#         Vs_surface = max(40, round((shallow_c * 1.05) / 40) * 40)
#         Vs_bottom  = max(Vs_surface + vs_step, round((deep_c * 1.05) / 40) * 40)
#
#         # === 阶梯数 ===
#         n_steps = max(1, num_layers // step_layers)
#
#         # === 阶梯 Vs ===
#         step_vs = np.linspace(Vs_surface, Vs_bottom, n_steps)
#
#         # 展开成 num_layers 层
#         target_vs = np.repeat(step_vs, step_layers)
#         target_vs = target_vs[:num_layers]   # 截到指定层数
#
#         # === Vs_min / Vs_max ===
#         Vs_min = (target_vs * 0.8).astype(int)
#         Vs_max = (target_vs * 1.7).astype(int)
#
#         # 四舍五入到 50 的整数倍
#         Vs_min = np.round(Vs_min / 40) * 40
#         Vs_max = np.round(Vs_max / 40) * 40
#
#         # 最小为 50
#         Vs_min = np.maximum(Vs_min, 40)
#         Vs_max = np.maximum(Vs_max, 40)
#
#         # 保证单调递增
#         for i in range(1, num_layers):
#             Vs_min[i] = max(Vs_min[i], Vs_min[i-1])
#             Vs_max[i] = max(Vs_max[i], Vs_max[i-1])
#
#         # thickness = make_thickness(
#         #     num_layers=num_layers,
#         #     max_depth=max_depth,
#         #     mode=thk_mode,  # "piecewise_exp" or "exp"
#         #     r1=r1,
#         #     r2=r2,
#         #     ratio=ratio,
#         #     exp_r=exp_r,
#         #     integer_output=integer_thk  # 整数厚度控制
#         # )
#         thickness = np.full(
#             num_layers,
#             max_depth / num_layers,
#             dtype=float
#         )
#         # H_min / H_max / H
#         H_min = thickness
#         H_max = thickness
#         H = (H_min + H_max) / 2000
#
#         # # === 厚度均分 ===
#         # thickness = max_depth / num_layers
#         # H_min = np.full(num_layers, float(thickness * 1))
#         # H_max = np.full(num_layers, float(thickness * 1))
#         # H = (H_min + H_max) / 2000
#
#         models[key] = np.vstack([Vs_min, Vs_max, H, H]).T.astype(float)
#
#     return models


# def make_models(data, max_depth):
#     models = {}
#     for k, arr in data.items():
#         c = np.asarray(arr)[:, 1]
#         # shallow = high-frequency portion (min c); deep = low-frequency (max c)
#         shallow_c = float(np.min(c))
#         deep_c = float(np.max(c))
#
#         # estimate Vs (scale factor 1.05), ensure sensible ordering
#         Vs_shallow = max(50.0, shallow_c * 1.05)
#         Vs_deep = max(Vs_shallow + 50.0, deep_c * 1.05)
#
#         # target profile (20 layers)
#         target = np.linspace(Vs_shallow, Vs_deep, 20)
#
#         # Vs bounds ±15% around target, floored/ceiled to int
#         Vs_min = np.maximum(50, np.floor(target * 0.6).astype(int))
#         Vs_max = np.ceil(target * 1.4).astype(int)
#
#         # enforce non-decreasing bounds (monotonic)
#         for i in range(1, 20):
#             if Vs_min[i] < Vs_min[i - 1]:
#                 Vs_min[i] = Vs_min[i - 1]
#             if Vs_max[i] < Vs_max[i - 1]:
#                 Vs_max[i] = max(Vs_max[i], Vs_max[i - 1])
#
#         # thickness distribution: linear ramp then scale to sum = max_depth
#         base = np.linspace(1.0, 5.0, 20)
#         thickness = base / base.sum() * max_depth
#
#         H_min = np.maximum(1, np.floor(thickness * 0.8).astype(int))
#         H_max = np.maximum(H_min, np.ceil(thickness * 1.2).astype(int))
#
#         # avoid zeros
#         H_min[H_min == 0] = 1
#         H_max[H_max == 0] = 1
#         H = (H_max + H_min) / 2
#         models[k] = np.vstack([Vs_min, Vs_max, H, H]).T.astype(int)
#     return models


def init_model(in_path, dep):
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    models = make_models(data, dep)
    return models


def model_c(models, key):
    model_array = models[key]
    layer_defs = convert_model_to_layer_definitions(model_array)
    return layer_defs


def filter_small_arrays(data_dict):
    """
    删除字典中第一维长度 <=5 的 numpy 数组
    """
    new_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            # 检查是否是二维数组并且第一维长度>5
            if value.ndim == 2 and value.shape[0] > 5:
                new_dict[key] = value
        else:
            # 非数组的内容原样保留（如果你不想保留可以删掉这段）
            new_dict[key] = value
    return new_dict


def interpolate_dict_to_32(data_dict):
    """
    将 dict 中的每个 n×2 数组插值到 32×2，插值范围不超过原数组边界。
    """
    new_dict = {}
    for key, arr in data_dict.items():
        arr = np.asarray(arr)
        # 确保是 n×2 的数组
        if arr.ndim != 2 or arr.shape[1] != 2:
            print(f"Skipping key {key}: not an n×2 array")
            continue
        # 原始 x, y
        x = arr[:, 0]
        y = arr[:, 1]
        # 构造插值函数，不允许外推
        f = interp1d(x, y, kind="linear", bounds_error=True)
        # 新的 32 个点（严格在边界内）
        x_new = np.linspace(x.min(), x.max(), 12)
        # 插值后的 y
        y_new = f(x_new)
        # 合并成 32×2
        new_arr = np.column_stack([x_new, y_new])
        new_dict[key] = new_arr
    return new_dict


def interpolate_missing_keys(data_dict, start=10, end=2410, step=10):
    # 先确保所有 key 是 int
    data_dict = {int(k): np.asarray(v, dtype=float) for k, v in data_dict.items()}
    existing_keys = sorted(data_dict.keys())
    shape = data_dict[existing_keys[0]].shape
    # 创建完整 key 列表
    full_keys = list(range(start, end + step, step))
    # 堆叠数组 (K, N, M)
    stack = np.stack([data_dict[k] for k in existing_keys], axis=0)
    existing_keys_arr = np.array(existing_keys, dtype=float)
    full_keys_arr = np.array(full_keys, dtype=float)
    N, M = shape
    new_stack = np.zeros((len(full_keys_arr), N, M))
    for i in range(N):
        for j in range(M):
            y = stack[:, i, j]
            new_stack[:, i, j] = np.interp(full_keys_arr, existing_keys_arr, y)
    # 生成新 dict
    return {k: new_stack[idx] for idx, k in enumerate(full_keys)}
