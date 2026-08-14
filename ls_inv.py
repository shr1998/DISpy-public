# -*- coding: utf-8 -*-
# @Time : 2025/12/8 16:27
# @Site :
# @File : inv_ls.py
# @Software: PyCharm
import time

import numpy as np
# from _surf96 import dltar,surf96
from _surf96_np import dltar4_vector
from gsurf96 import surf96
from scipy.optimize import least_squares,minimize
from scipy.interpolate import interp1d

from threading import local

# 姣忎釜绾跨▼鏈夌嫭绔嬬殑瀛樺偍
thread_local = local()


class ResidualCallback:
    def __init__(self):
        self.r = None

    def __call__(self, r):
        self.r = r


def random_initial_models_monotonic_perturb(
    base_model,
    low_bound,
    up_bound,
    nstart,
    scale=0.15,
    min_inc=1e-3,
    drop_tol=8e-3,
    jump_prob=0.55,
    jump_scale=(0.04, 0.16),
    rng=None,
):
    """
    鍦?base_model 闄勮繎鍋?卤 鎵板姩
    骞朵繚璇佹ā鍨嬩弗鏍奸€掑
    """
    if rng is None:
        rng = np.random.default_rng()

    base = np.asarray(base_model, dtype=float)
    low = np.asarray(low_bound, dtype=float)
    up = np.asarray(up_bound, dtype=float)

    models = []

    for _ in range(nstart):
        # 1. 鍙屽悜鎵板姩
        noise = rng.normal(0.0, scale, size=base.shape)
        m = base * (1.0 + noise)

        # 1.5 inject a jump-style initial family to improve sharp-interface coverage
        if len(m) >= 4 and rng.random() < jump_prob:
            k = int(rng.integers(1, len(m) - 1))
            amp = float(rng.uniform(jump_scale[0], jump_scale[1]))
            m[k:] *= (1.0 + amp)

        # 2. bounds
        m = np.clip(m, low, up)

        # 3. near-monotonic safeguard:
        # allow a tiny local drop, only correct clear inversions
        for i in range(1, len(m)):
            if m[i] < (m[i - 1] - drop_tol):
                m[i] = min(
                    max(m[i - 1] - drop_tol + min_inc, low[i]),
                    up[i],
                )

        models.append(m)

    return models


def invert_surf96_adaptive_mc(
    initial_model,
    nd,
    period_dict,
    obs_dict,
    low_bound,
    up_bound,
    callback_plotdata,
    callback_plotdata1,
    ti,
    nstart=10,
    n_round=6,
    init_scale=0.3,
    scale_decay=0.8,
    inv_object="vs",
    d_low_bound=None,
    d_up_bound=None,
):
    if str(inv_object).lower() == "vsandd":
        return invert_surf96_vsandd_mc(
            initial_model=initial_model,
            initial_thickness=nd,
            period_dict=period_dict,
            obs_dict=obs_dict,
            low_bound=low_bound,
            up_bound=up_bound,
            d_low_bound=d_low_bound,
            d_up_bound=d_up_bound,
            callback_plotdata=callback_plotdata,
            callback_plotdata1=callback_plotdata1,
            ti=ti,
            nstart=nstart,
            n_round=n_round,
            init_scale=init_scale,
            scale_decay=scale_decay,
            inv_object=inv_object,
        )

    rng = np.random.default_rng()

    res_cb = ResidualCallback()

    best = {
        "loss": np.inf,
        "model": initial_model.copy(),
        "res": None,
        "thickness": None,
        "inv_object": inv_object,
    }
    base_model = initial_model.copy()
    scale = init_scale
    # thickness = np.full(20, 4.0)  # 姣忓眰鍘氬害 m
    # vs = np.array([240] * 5 + [420] * 2 + [550] * 2 + [720] * 2 + [950] * 3 + [1180] * 2 + [1250, 1300, 1320, 1320])
    #
    # # model = np.column_stack([thickness / 1000, vs])  # 甯歌鏍煎紡锛歔鍘氬害(km), Vs(m/s)]
    # # 鎴栨洿瀹屾暣锛歔鍘氬害(km), Vs, Vp(鍙€?, rho(鍙€?]
    # base_model = vs/1000
    # nd = thickness/1000
    for iround in range(n_round):
        obs_dict_t= dict([next(reversed(obs_dict.items()))])
        period_dict_t = dict([next(reversed(period_dict.items()))])
        # period_dict_t = dict([next(reversed(period_dict.items()))])
        # print(f"\n[ROUND {iround+1}/{n_round}] scale={scale:.3f}")

        # === 1. 鍥寸粫褰撳墠 best 鐢熸垚闅忔満妯″瀷 ===
        init_models = random_initial_models_monotonic_perturb(
            base_model,
            low_bound,
            up_bound,
            nstart,
            scale,
            rng=rng,
        )

        improved = False

        for i, x0 in enumerate(init_models):
            # print(f"[MC] round {iround} start {i+1}/{nstart}")
            # try:
            res = least_squares(
                misfit_multimode,
                x0=x0,
                args=(nd, period_dict_t, obs_dict_t,
                      callback_plotdata, ti, res_cb),  # 鉂?MC 闃舵涓嶇敾鍥?                    bounds=(low_bound, up_bound),
                method="dogbox",
                diff_step=1e-2,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=1500,
                verbose=0,
            )

                # print(f'init model{x0}')
                # res = minimize(
                #     scalar_loss,
                #     x0=x0,
                #     args=(nd, period_dict, obs_dict),
                #     method="L-BFGS-B",
                #     bounds=list(zip(low_bound, up_bound)),
                #     options={
                #         "maxiter": 500,
                #         "ftol": 1e-12,
                #         "gtol": 1e-8,
                #         "disp": False,
                #     },
                # )
            # except Exception as e:
            #     print("[MC] failed:", e)
            #     continue

            loss = 0.5 * np.dot(res.fun, res.fun)
            # loss = res.fun
            # print(f"[MC] loss = {loss:.4e}")

            if loss < best["loss"]:
                best.update(
                    loss=loss,
                    model=res.x.copy(),
                    res=res,
                )
                base_model = res.x.copy()
                improved = True

                print(
                    f"[BEST] round={iround} loss={loss:.4e}\n"
                    f"model={res.x}"
                )
                # === 鍙湪鐪熸鎻愬崌鏃舵洿鏂?UI ===
                callback_plotdata(res.x, ti)
                callback_plotdata1(ti)
            # elif isinstance(res_cb.r, np.ndarray):
            #     c_1 = obs_dict_t[0]
            #     means = adaptive_sampling_by_diff(c_1, res_cb.r, len(res.x))
            #     base_model = base_model * (1 + means)


        # === 2. 濡傛灉杩欎竴杞病鏈夋敼杩涳紝鎻愬墠缁撴潫 ===
        if not improved:
            # break
            print("[STEP] no improvement in this round")
            if len(obs_dict)==1:
                break
            base_model_h = base_model.copy()
            best["loss"] = 1
            for iround_h in range(n_round):
                # print(f"\n[ROUND {iround_h+iround + 1}/{n_round+n_round}] scale={scale*0.5:.3f}")

                # === 1. 鍥寸粫褰撳墠 best 鐢熸垚闅忔満妯″瀷 ===
                init_models = random_initial_models_monotonic_perturb(
                    base_model_h,
                    low_bound,
                    up_bound,
                    nstart,
                    scale*0.5,
                    rng=rng,
                )

                improved = False

                for i, x0 in enumerate(init_models):
                    # print(f"[MC] round {iround_h+iround} start {i + 1}/{nstart}")

                    try:
                        res = least_squares(
                            misfit_multimode,
                            x0=x0,
                            args=(nd, period_dict, obs_dict,
                                  callback_plotdata, ti, res_cb),  # 鉂?MC 闃舵涓嶇敾鍥?                            bounds=(low_bound, up_bound),
                            method="trf",
                            tr_solver='lsmr',
                            diff_step=1e-2,
                            ftol=1e-10,
                            xtol=1e-10,
                            gtol=1e-10,
                            max_nfev=3000,
                            verbose=0,
                        )

                    except Exception as e:
                        print("[MC] failed:", e)
                        continue

                    loss = 0.5 * np.dot(res.fun, res.fun)
                    # loss = res.fun
                    # print(f"[MC] loss = {loss:.4e}")

                    if loss < best["loss"]:
                        best.update(
                            loss=loss,
                            model=res.x.copy(),
                            res=res,
                        )
                        base_model_h = res.x.copy()
                        improved = True

                        print(
                            f"[BEST] round={iround} loss={loss:.4e}\n"
                            f"model={res.x}"
                        )

                        # === 鍙湪鐪熸鎻愬崌鏃舵洿鏂?UI ===
                        callback_plotdata(res.x, ti)
                        callback_plotdata1(ti)

            if not improved:
                print("[STOP] no improvement in this round")
                break
            break

        # === 3. 缂╁皬鎵板姩灏哄害 ===
        scale *= scale_decay

    # Multi-mode only refinement:
    # keep single-mode path unchanged, and only refine when there are >1 modes.
    if len(obs_dict) > 1 and best.get("model") is not None:
        mode_keys = sorted(list(obs_dict.keys()))
        base_model_mm = np.asarray(best["model"], dtype=float).copy()
        mm_nstart = max(2, int(nstart // 2))
        mm_scale = max(float(scale) * 0.5, float(init_scale) * 0.25)

        for stage_size in range(2, len(mode_keys) + 1):
            active_modes = mode_keys[:stage_size]
            obs_stage = {k: obs_dict[k] for k in active_modes}
            period_stage = {k: period_dict[k] for k in active_modes}
            stage_improved = False

            init_models = random_initial_models_monotonic_perturb(
                base_model_mm,
                low_bound,
                up_bound,
                mm_nstart,
                mm_scale,
                rng=rng,
            )

            for x0 in init_models:
                try:
                    res = least_squares(
                        misfit_multimode,
                        x0=x0,
                        args=(nd, period_stage, obs_stage, callback_plotdata, ti, res_cb),
                        bounds=(low_bound, up_bound),
                        method="trf",
                        tr_solver="lsmr",
                        diff_step=1e-2,
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-10,
                        max_nfev=2000,
                        verbose=0,
                    )
                except Exception as e:
                    print("[MM REFINE] failed:", e)
                    continue

                loss = 0.5 * np.dot(res.fun, res.fun)
                if loss < best["loss"]:
                    best.update(
                        loss=loss,
                        model=res.x.copy(),
                        res=res,
                    )
                    base_model_mm = res.x.copy()
                    stage_improved = True
                    print(
                        f"[MM REFINE] stage={stage_size} "
                        f"modes={active_modes} loss={loss:.4e}"
                    )
                    callback_plotdata(res.x, ti)
                    callback_plotdata1(ti)

            # keep trying next stage even when one stage has no gain
            mm_scale *= 0.85
            if not stage_improved:
                print(f"[MM REFINE] stage={stage_size} no improvement")

    return best


def _forward_vsandd(vs_model, thickness_model, period_dict):
    H = np.maximum(np.asarray(thickness_model, dtype=float), 1.0e-4)
    Vs = np.asarray(vs_model, dtype=float)
    Vp = Vs * 1.73
    rho = 0.54 * Vp + 0.25
    result = {}
    for mode_id, T in period_dict.items():
        T_arr = np.asarray(T, dtype=float)
        try:
            c = surf96(
                T_arr,
                H,
                Vp,
                Vs,
                rho,
                itype=0,
                mode=mode_id,
            )
            result[mode_id] = np.asarray(c, dtype=float)
        except Exception:
            result[mode_id] = np.full_like(T_arr, np.nan, dtype=float)
    return result


def _misfit_multimode_vsandd(
    model_params,
    n_vs,
    period_dict,
    obs_dict,
    rec_cb,
    eps=1e-8,
):
    vs_model = np.asarray(model_params[:n_vs], dtype=float)
    thickness_model = np.asarray(model_params[n_vs:], dtype=float)

    pred = _forward_vsandd(vs_model, thickness_model, period_dict)
    residuals = []

    for mode_id, c_obs in obs_dict.items():
        if mode_id not in pred:
            continue
        c_pre = np.asarray(pred[mode_id], dtype=float)
        c_obs = np.asarray(c_obs, dtype=float)
        if c_pre.shape != c_obs.shape:
            c_pre = np.full_like(c_obs, np.nan, dtype=float)
        if (not np.all(np.isfinite(c_pre))) or np.any(c_pre <= 0):
            residuals.append(np.full_like(c_obs, 10.0, dtype=float))
            continue
        if np.sum(c_pre - c_obs) >= 0:
            r = np.abs(c_pre - c_obs) / (c_obs + eps)
        else:
            r = np.abs(c_pre - c_obs) / (np.maximum(c_pre, eps))
        if mode_id == 0:
            r1 = (c_pre - c_obs) / (c_pre + eps)
        r = np.expm1(np.clip(r, 0.0, 20.0))
        r *= (1.0 / (1.0 + 0.5 * max(0, int(mode_id))))
        residuals.append(r)

    # Vs soft monotonic (allow small drop)
    if vs_model.size >= 2:
        dvs = np.diff(vs_model)
        mono_viol = np.maximum(0.0, -(dvs + 8e-3))
        if mono_viol.size > 0:
            residuals.append(0.15 * (mono_viol / 0.02))

    # Thickness smoothness/stability (soft)
    if thickness_model.size >= 2:
        dstep = np.diff(thickness_model)
        residuals.append(0.06 * (dstep / (np.mean(np.abs(thickness_model)) + eps)))

    if 0 in obs_dict:
        try:
            rec_cb(np.array(r1))
        except Exception:
            pass

    if len(residuals) == 0:
        return np.array([1.0], dtype=float)
    return np.hstack(residuals)


def invert_surf96_vsandd_mc(
    initial_model,
    initial_thickness,
    period_dict,
    obs_dict,
    low_bound,
    up_bound,
    d_low_bound,
    d_up_bound,
    callback_plotdata,
    callback_plotdata1,
    ti,
    nstart=10,
    n_round=6,
    init_scale=0.3,
    scale_decay=0.8,
    inv_object="vsandd",
):
    rng = np.random.default_rng()
    res_cb = ResidualCallback()

    vs0 = np.asarray(initial_model, dtype=float).copy()
    d0 = np.asarray(initial_thickness, dtype=float).copy()
    lb_vs = np.asarray(low_bound, dtype=float)
    ub_vs = np.asarray(up_bound, dtype=float)
    if d_low_bound is None:
        d_low_bound = np.maximum(d0 * 0.5, 1e-4)
    if d_up_bound is None:
        d_up_bound = d0 * 1.8
    lb_d = np.asarray(d_low_bound, dtype=float)
    ub_d = np.asarray(d_up_bound, dtype=float)

    x_best = np.concatenate([vs0, d0])
    lb = np.concatenate([lb_vs, lb_d])
    ub = np.concatenate([ub_vs, ub_d])
    n_vs = len(vs0)
    best_loss = np.inf
    best_res = None
    cur_scale = float(init_scale)

    for _ in range(max(1, int(n_round))):
        improved = False
        init_vs_list = random_initial_models_monotonic_perturb(
            x_best[:n_vs], lb_vs, ub_vs, nstart, scale=cur_scale, rng=rng
        )
        for vs_seed in init_vs_list:
            d_noise = rng.normal(0.0, max(0.03, cur_scale * 0.2), size=d0.shape)
            d_seed = np.clip(x_best[n_vs:] * (1.0 + d_noise), lb_d, ub_d)
            x0 = np.concatenate([vs_seed, d_seed])
            try:
                res = least_squares(
                    _misfit_multimode_vsandd,
                    x0=x0,
                    args=(n_vs, period_dict, obs_dict, res_cb),
                    bounds=(lb, ub),
                    method="trf",
                    tr_solver="lsmr",
                    diff_step=1e-2,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=2500,
                    verbose=0,
                )
            except Exception as e:
                print("[VSANDD] failed:", e)
                continue

            loss = 0.5 * np.dot(res.fun, res.fun)
            if loss < best_loss:
                best_loss = loss
                best_res = res
                x_best = np.asarray(res.x, dtype=float).copy()
                improved = True
                callback_plotdata(x_best[:n_vs], ti)
                callback_plotdata1(ti)
                print(f"[VSANDD BEST] task={ti} loss={loss:.4e}")

        if not improved:
            break
        cur_scale *= float(scale_decay)

    return {
        "loss": best_loss if np.isfinite(best_loss) else np.inf,
        "model": x_best[:n_vs].copy(),
        "thickness": x_best[n_vs:].copy(),
        "res": best_res,
        "inv_object": inv_object,
    }


# def invert_surf96_multistart(
#     initial_model,
#     nd,
#     period_dict,
#     obs_dict,
#     low_bound,
#     up_bound,
#     callback_plotdata,
#     ti,
#     nstart=10,
# ):
#     best = {
#         "loss": np.inf,
#         "model": None,
#         "res": None,
#         "iter": -1,
#     }
#
#     # 鐢熸垚闅忔満鍒濆妯″瀷
#     init_models = random_initial_models_monotonic_perturb(
#         initial_model,
#         low_bound,
#         up_bound,
#         nstart,
#     )
#
#     for i, x0 in enumerate(init_models):
#         print(f"[MC] start {i+1}/{nstart}")
#
#         try:
#             res = least_squares(
#                 misfit,
#                 x0=x0,
#                 args=(nd, period_dict, obs_dict,
#                       callback_plotdata, ti),
#                 bounds=(low_bound, up_bound),
#                 method="dogbox",
#                 diff_step=1e-2,
#                 ftol=1e-12,
#                 xtol=1e-12,
#                 gtol=1e-12,
#                 max_nfev=2000,
#                 verbose=0,  # 鈽?鍏抽棴鍐呴儴鍒峰睆
#             )
#         except Exception as e:
#             print(f"[MC] start {i} failed:", e)
#             continue
#
#         loss = 0.5 * np.dot(res.fun, res.fun)
#
#         print(f"[MC] start {i} loss = {loss:.4e}")
#
#         if loss < best["loss"]:
#             best.update(
#                 loss=loss,
#                 model=res.x.copy(),
#                 res=res,
#                 iter=i,
#             )
#
#             print(
#                 f"[BEST] start={i} loss={loss:.4e}\n"
#                 f"model={res.x}"
#             )
#
#             # 鉁?鍙湁鈥滄渶濂界粨鏋溾€濇墠鏇存柊 UI
#             callback_plotdata(res.x, int(ti))
#
#     return best


def get_best_state():
    if not hasattr(thread_local, 'best_state'):
        thread_local.best_state = {
            "loss": float('inf'),
            "model": None,
            "iter": 0,
        }
    return thread_local.best_state


def load_v(grd):
    v = np.array([
        [5.0, 7.00, 3.50, 2.00],
        [5.0, 7.00, 3.50, 2.00],
        [10.0, 7.00, 3.50, 2.00],
        [10.0, 7.60, 3.80, 2.00],
        [10.0, 8.40, 4.20, 2.00],
        [10.0, 9.00, 4.50, 2.00],
        [10.0, 9.40, 4.70, 2.00],
        [10.0, 9.60, 4.80, 2.00],
        [10.0, 9.60, 4.80, 2.00],
    ])
    return v
grd = []
velocity_model = load_v(grd)
velocity_model_thickness=velocity_model[:,0]
velocity_model_vp=velocity_model[:,1]
velocity_model_vs=velocity_model[:,2]
velocity_model_rho=velocity_model[:,3]
best_state = {
    "loss": np.inf,
    "model": None,
    "iter": 0,
}


def _warmup_surf96_safe():
    dummy_model = {
        'period': np.logspace(0.5, 2, 20),
        'thickness': np.array([3.0, 10.0, 20.0]),
        'vp': np.array([5.5, 6.8, 7.8, 8.2]),
        'vs': np.array([3.1, 3.9, 4.5, 4.7]),
        'rho': np.array([2.5, 2.8, 3.1, 3.3]),
    }
    try:
        surf96(**dummy_model, mode=0, itype=0, ifunc=3)
    except:
        pass  # 姘歌繙涓嶈棰勭儹褰卞搷涓荤▼搴?
# _warmup_surf96_safe()


def forward_p(model_params, nd, period_dict, obs_dict):
    H = np.maximum(np.asarray(nd, dtype=float), 1.0e-4)
    Vs = model_params

    # auto compute Vp and density
    Vp = Vs * 1.73
    rho = (0.32 * Vp + 0.77)
    c = obs_dict[0]
    t = period_dict[0]
    H = np.expand_dims(H, axis=0)
    H = H
    Vs = np.expand_dims(Vs, axis=0)
    Vp = np.expand_dims(Vp, axis=0)
    rho = np.expand_dims(rho, axis=0)
    rho = rho
    c = np.expand_dims(c, axis=0)
    t = np.expand_dims(t, axis=0)

    try:
        F = dltar4_vector(c, t, H, Vp, Vs, rho, llw=-1)
    except Exception:
        F = np.full_like(c, np.nan, dtype=float)

    return F


def forward(model_params, nd,period_dict):
    H = np.maximum(np.asarray(nd, dtype=float), 1.0e-4)
    Vs = model_params

    # auto compute Vp and density
    Vp = Vs * 1.73
    rho = (0.54 * Vp + 0.25)
    # 閫愰樁璁＄畻鐩搁€熷害
    result = {}

    for mode_id, T in period_dict.items():
        T_arr = np.asarray(T, dtype=float)
        try:
            c = surf96(
                T_arr,               # period array
                H,
                Vp,
                Vs,
                rho,
                itype=0,         # 鐩搁€熷害
                mode=mode_id,   # 闃舵锛屽 0, 1, 2
            )
            result[mode_id] = np.asarray(c, dtype=float)
        except Exception:
            result[mode_id] = np.full_like(T_arr, np.nan, dtype=float)

    return result


def misfit_p(model_params, nd, period_dict, obs_dict):
    pred = forward_p(model_params, nd, period_dict, obs_dict)
    # F = 1e-1 ** abs(pred_dict) - 1
    # F = pred_dict / (np.max(pred_dict, axis=1, keepdims=True) - np.min(pred_dict, axis=1, keepdims=True))
    F = 1e-1 ** abs(pred) - 1
    F = np.sum(abs(F), axis=1)
    return F


def misfit_multimode(
    model_params,
    nd,
    period_dict,
    obs_dict,
    callback_plotdata,
    ti,
    rec_cb,
    delta=0.1,
    eps=1e-8,
):
    """
    澶氶樁棰戞暎 residual锛堝伐绋嬪瀷锛岄€傜敤浜?MC / adaptive MC锛?
    杩斿洖鍊硷細
        1D residual 鍚戦噺锛堟墍鏈?mode 鎷兼帴锛?    """

    best = get_best_state()
    pred = forward(model_params, nd, period_dict)

    residuals = []
    total_loss = 0.0

    mode_losses = {}
    for mode_id, c_obs in obs_dict.items():

        if mode_id not in pred:
            continue

        c_pre = np.asarray(pred[mode_id], dtype=float).copy()
        c_obs = np.asarray(c_obs, dtype=float)
        if c_pre.shape != c_obs.shape:
            c_pre = np.full_like(c_obs, np.nan, dtype=float)
        if (not np.all(np.isfinite(c_pre))) or np.any(c_pre <= 0):
            bad_r = np.full_like(c_obs, 10.0, dtype=float)
            residuals.append(bad_r)
            mode_loss = float(np.sum(bad_r))
            mode_losses[mode_id] = mode_loss
            total_loss += mode_loss
            continue

        # surf96 寮傚父鍏滃簳
        # if mode_id!=0:
        #     c_pre[c_pre <= 0] = 1.5 * pred[mode_id-1][c_pre <= 0].copy()

        # relative residual
        if np.sum(c_pre - c_obs) >= 0:
            r = np.abs(c_pre - c_obs) / (c_obs + eps)
        else:
            r = np.abs(c_pre - c_obs) / (np.maximum(c_pre, eps))

        # 鍙褰曞熀闃讹紝鐢ㄤ簬鍥炶皟
        if mode_id == 0 and 0 in obs_dict:
            r1 = (c_pre - c_obs) / (c_pre + eps)

        # nonlinear amplify with clip to avoid overflow / non-finite residuals
        r = np.expm1(np.clip(r, 0.0, 20.0))

        # mode weight (safe, monotonic decay by mode index)
        r *= (1.0 / (1.0 + 0.5 * max(0, int(mode_id))))

        residuals.append(r)
        mode_loss = np.sum(r)
        mode_losses[mode_id] = mode_loss
        total_loss += mode_loss

    # ----- soft model regularization for stable, mostly-increasing Vs -----
    # keep the model mostly monotonic, but allow tiny local drops/jumps.
    vs = np.asarray(model_params, dtype=float)
    reg_terms = []
    if vs.size >= 2:
        dvs = np.diff(vs)
        allow_drop = 8e-3  # km/s, allow small local inversion
        mono_viol = np.maximum(0.0, -(dvs + allow_drop))
        if mono_viol.size > 0:
            reg_terms.append(0.12 * (mono_viol / 0.02))

    if vs.size >= 3:
        dd = np.diff(vs, n=2)
        curv_tol = 7.0e-2  # allow sharper jumps, suppress only strong oscillations
        curv_viol = np.maximum(0.0, np.abs(dd) - curv_tol)
        if curv_viol.size > 0:
            reg_terms.append(0.025 * (curv_viol / 0.03))

    if reg_terms:
        reg_res = np.hstack(reg_terms)
        residuals.append(reg_res)
        total_loss += float(np.sum(np.abs(reg_res)))
    # high mode threshold
    if 0 in mode_losses:
        base_loss = mode_losses[0]
        high_mode_threshold = 1.8 * base_loss  # 鎴?2.0 * base_loss
    else:
        high_mode_threshold = 2.0

    high_modes_ok = all(mode_losses.get(m, 0) < high_mode_threshold for m in obs_dict if m > 0)
    # best-state 鏇存柊锛堜笌浣犲師鏉ヤ竴鑷达級
    if total_loss < best["loss"] and high_modes_ok:
        best["loss"] = total_loss
        best["model"] = model_params.copy()
        best["iter"] += 1

        # callback_plotdata(model_params, ti)
        if 0 in obs_dict:
            rec_cb(np.array(r1))

    return np.hstack(residuals)


def misfit(model_params, nd, period_dict, obs_dict,
           callback_plotdata, ti,rec_cb,
           delta=0.1, eps=1e-8):

    # global r1
    best = get_best_state()
    pred = forward(model_params, nd, period_dict)

    residuals = []

    for mode_id, c_obs in obs_dict.items():
        c_pre = pred[mode_id]
        c_pre[c_pre<=0] = 3
        # 1. 鐩稿璇樊
        if np.sum(c_pre - c_obs) >= 0:
            r = abs(c_pre - c_obs) / c_obs
        else:
            r = abs(c_pre - c_obs) / c_pre
        if mode_id == 0 :
            r1 = (c_pre - c_obs) / c_pre
        # 2. mode 鍐?RMS 褰掍竴鍖栵紙MC & 姊害閮界ǔ瀹氾級
        # if (abs(r) > 0.01).any():
        r = np.exp(r)-1
        # else:
        #     r = np.log(r)
        #     r = (r + 9/r.max()+9)
        # rms = np.sqrt(np.mean(r**2)) + eps
        # r /= rms
        # 3. soft L1锛堝钩婊?Huber锛?        # r = np.sqrt(r**2 + delta**2) - delta

        residuals.append(r)
    # if np.sum(np.hstack(residuals)) < best_state["loss"]:
    #     best_state["loss"] = np.sum(np.hstack(residuals))
    #     best_state["model"] = model_params.copy()
    #     callback_plotdata1(pred[0], ti)
    #     callback_plotdata(model_params, ti)
    if np.sum(np.hstack(residuals)) < best["loss"]:
        best["loss"] = np.sum(np.hstack(residuals))
        best["model"] = model_params.copy()
        best["iter"] += 1
        # callback_plotdata(model_params, ti)
        rec_cb(np.array(r1))

    return np.hstack(residuals)


def misfit_l(model_params, nd, period_dict, obs_dict, eps=1e-8):

    pred = forward(model_params, nd, period_dict)

    residuals = []
    for mode_id, c_obs in obs_dict.items():
        c_pre = pred[mode_id]

        # 鐩稿璇樊锛堣繖鏄?OK 鐨勶級
        r = (c_pre - c_obs) / (c_obs + eps)
        # 2. mode 鍐?RMS 褰掍竴鍖栵紙MC & 姊害閮界ǔ瀹氾級
        # rms = np.sqrt(np.mean(r**2)) + eps
        # r /= rms

        residuals.append(r)

    return np.hstack(residuals)


def scalar_loss(x, nd, period_dict, obs_dict,
                delta=0.1, eps=1e-8):

    r = misfit_l(x, nd, period_dict, obs_dict, eps)

    # mode / 鍏ㄥ眬 RMS 褰掍竴鍖栵紙鏀惧湪 loss 灞傦級
    rms = np.sqrt(np.mean(r**2)) + eps
    r = r / rms

    # soft-L1 / pseudo-Huber loss
    loss = np.sum(delta**2 * (np.sqrt(1 + (r/delta)**2) - 1))

    return loss


def make_callback(nd, period_dict, obs_dict, ti,
                  callback_plotdata,
                  callback_plotdata1):
    best = get_best_state()
    def callback(xk):
        best["iter"] += 1

        loss = scalar_loss(xk, nd, period_dict, obs_dict)

        if loss < best["loss"]:
            best["loss"] = loss
            best["model"] = xk.copy()
            # ===== 浣犵殑瀹炴椂缁樺浘 =====
            callback_plotdata(xk, ti)
            # callback_plotdata1(pred[0], ti)

        print(
            f"[BEST] iter={best['iter']:4d} "
            f"loss={loss:.4e}"
            f"model={xk}"
        )

    return callback


# def invert_surf96(initial_model, nd, period_dict, obs_dict, low_bound, ub_bound, callback_plotdata, callback_plotdata1, ti):
#     best_state = {
#         "loss": np.inf,
#         "model": None,
#         "iter": 0,
#     }
#     # res = least_squares(
#     #     misfit,
#     #     x0=initial_model,
#     #     args=(nd, period_dict, obs_dict, callback_plotdata, callback_plotdata1, ti),
#     #     bounds=(low_bound, ub_bound),
#     #     method="dogbox",
#     #     diff_step=1e-3,
#     #     ftol=1e-12,
#     #     xtol=1e-12,
#     #     gtol=1e-12,
#     #     max_nfev=2000,
#     #     verbose=2,
#     #
#     # )
#     # print("\n=== Inversion Result ===")
#     # print("success =", res.success)
#     # print("best model =", res.x)
#     callback = make_callback(nd, period_dict, obs_dict, ti)
#     res = minimize(
#         scalar_loss,
#         initial_model,
#         args=(nd, period_dict, obs_dict,
#               callback_plotdata, callback_plotdata1, ti),
#         method="L-BFGS-B",
#         bounds=list(zip(low_bound, ub_bound)),
#         options=dict(
#             maxiter=300,
#             ftol=1e-12,
#             gtol=1e-6,
#         ),
#         callback=callback
#
#     )
#     return res

def invert_surf96(
    initial_model,
    nd,
    period_dict,
    obs_dict,
    low_bound,
    up_bound,
    callback_plotdata,
    ti,
):
    def identify_resolvable_layers(
            x0,
            loss_fn,
            nd, period_dict, obs_dict,
            eps=1e-3,
            grad_tol=1e-4
    ):
        """
        杩斿洖锛?            resolvable_mask: bool array, True=鍙垎杈?            grad: 鏁板€兼搴?        """
        n = len(x0)
        grad = np.zeros(n)

        f0 = loss_fn(x0, nd, period_dict, obs_dict)

        for i in range(n):
            x1 = x0.copy()
            x2 = x0.copy()
            x1[i] += eps
            x2[i] -= eps

            f1 = loss_fn(x1, nd, period_dict, obs_dict)
            f2 = loss_fn(x2, nd, period_dict, obs_dict)

            grad[i] = (f1 - f2) / (2 * eps)

        resolvable_mask = np.abs(grad) > grad_tol
        return resolvable_mask, grad
    if not best_state['model'] is None:
        mask, grad = identify_resolvable_layers(
            best_state['model'],  # MC 鎴栧墠涓€闃舵鏈€浼樻ā鍨?            scalar_loss,
            nd, period_dict, obs_dict
        )

        print("Layer | grad | resolvable")
        for i, (g, m) in enumerate(zip(grad, mask)):
            print(f"{i:2d} | {g: .2e} | {m}")
    res = least_squares(
        misfit,
        x0=initial_model,
        args=(nd, period_dict, obs_dict, callback_plotdata, ti),
        bounds=(low_bound, up_bound),
        method="dogbox",
        diff_step=1e-2,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-6,
        max_nfev=2000,
        verbose=2,

    )
    # callback = make_callback(
    #     nd,
    #     period_dict,
    #     obs_dict,
    #     ti,
    #     callback_plotdata,
    #     callback_plotdata1,
    # )
    # print(f'init model{initial_model}')
    # res = minimize(
    #     scalar_loss,
    #     x0=initial_model,
    #     args=(nd, period_dict, obs_dict),
    #     method="L-BFGS-B",
    #     bounds=list(zip(low_bound, up_bound)),
    #     options={
    #         "maxiter": 500,
    #         "ftol": 1e-12,
    #         "gtol": 1e-8,
    #         "disp": False,
    #     },
    #     callback=callback,
    # )

    return res


def curve_misfit_xy(obs_phase_vel, syn_phase_vel,kind="rmse"):
    """
    鑷姩鎻掑€煎埌瑙傛祴鏁版嵁鐐癸紝鐒跺悗璁＄畻璇樊
    """
    # 灏嗗悎鎴愭洸绾挎彃鍊煎埌瑙傛祴棰戠巼鐐?    x_obs, y_obs = obs_phase_vel
    x_syn, y_syn = syn_phase_vel
    y_syn_interp = np.interp(x_obs, x_syn, y_syn)

    residual = y_obs - y_syn_interp

    if kind == "rmse":
        return np.sqrt(np.mean(residual**2))
    elif kind == "mse":
        return np.mean(residual**2)
    elif kind == "l2":
        return np.linalg.norm(residual)
    else:
        raise ValueError("kind must be 'rmse', 'mse', or 'l2'")


def finite_difference_jacobian(t, thickness, vp, vs, rho, param_name='vs', delta=1e-3):
    """
    璁＄畻闆呭彲姣旂煩闃碉紝閽堝vs鍙傛暟
    t: 棰戠巼鐐规暟缁?    thickness, vp, vs, rho: 褰撳墠妯″瀷鍙傛暟鏁扮粍
    param_name: 褰撳墠鍙敮鎸?vs'锛屽彲鎵╁睍
    杩斿洖锛欽acobian鐭╅樀锛宻hape=(len(t), len(vs))
    """
    n_params = len(vs)
    n_freq = len(t)
    J = np.zeros((n_freq, n_params))

    base_disp = surf96(t, thickness, vp, vs, rho)

    for i in range(n_params):
        vs_perturbed = vs.copy()
        vs_perturbed[i] += delta
        disp_perturbed = surf96(t, thickness, vp, vs_perturbed, rho)
        J[:, i] = (disp_perturbed - base_disp) / delta

    return J


def least_squares_inversion(t, obs_phase_vel, thickness, vp, vs, rho,
                            max_iter=10, tol=1e-3):
    """
    鍩轰簬鏈€灏忎簩涔樼殑鍙嶆紨锛岄拡瀵箆s鍙傛暟
    """
    vs_current = vs.copy()

    for iter_num in range(max_iter):
        syn_phase_vel = surf96(t, thickness, vp, vs_current, rho)
        residual = curve_misfit_xy(obs_phase_vel, syn_phase_vel)

        cost = np.sum(residual**2)
        print(f"Iter {iter_num+1}, cost: {cost:.6f}")

        if cost < tol:
            print("鏀舵暃")
            break

        J = finite_difference_jacobian(t, thickness, vp, vs_current, rho)

        JTJ = J.T @ J
        JTres = J.T @ residual

        try:
            delta_vs = np.linalg.solve(JTJ, JTres)
        except np.linalg.LinAlgError:
            print("linear solve failed, stop iteration")
            break

        vs_current += delta_vs

    return vs_current


def adaptive_sampling_by_diff(ref_curve, target_array, M):
    ref_curve = np.asarray(ref_curve)
    target_array = np.asarray(target_array)
    N = len(ref_curve)
    assert len(target_array) == N

    # 1. 鍏堝潎鍖€鍒?M 娈碉紝璁＄畻鍙樺寲閫熺巼锛堢粷瀵瑰樊鍒嗗拰锛?    indices = np.linspace(0, N, M + 1, dtype=int)
    diff = np.abs(np.diff(ref_curve, prepend=ref_curve[0]))
    segment_diff_sums = []
    for i in range(M):
        start, end = indices[i], indices[i+1]
        segment_diff_sums.append(diff[start:end].sum())
    segment_diff_sums = np.array(segment_diff_sums)

    # 2. 鏍规嵁鍙樺寲閫熺巼鍒嗛厤閲囨牱鐐逛釜鏁帮紝鍙樺寲閫熺巼澶у垎寰楀锛屾瘮渚嬪垎閰?    total_points = N
    weights = segment_diff_sums / segment_diff_sums.sum()
    counts = np.floor(weights * total_points).astype(int)
    counts[counts==0] = 1
    # 3. 淇鍥犲悜涓嬪彇鏁撮€犳垚鐨勭偣鏁扮己澶憋紝琛ュ埌鏈€澶iff娈?    diff_points_missing = total_points - counts.sum()
    counts[np.argmax(segment_diff_sums)] += diff_points_missing

    # 4. 鏍规嵁counts鏁扮粍鍔ㄦ€佸垏鍒唗arget_array锛岃绠楀潎鍊?    means = []
    start_idx = 0
    groups = []
    for count in counts:
        end_idx = start_idx + count
        if end_idx > N:
            end_idx = N
        groups.append((start_idx, end_idx))
        means.append(target_array[start_idx:end_idx].mean())
        start_idx = end_idx

    return np.array(means)

# module test block removed to keep runtime import-safe
