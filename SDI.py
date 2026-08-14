# @Time : 2024/6/11 10:11
# @File : SDI.py
import matplotlib.pyplot as plt
import numpy as np
import torch

# import utillis_new_test_1
from _surf96_vector_gpu import dltar4_vector
from model_init import monte_carlo_sampling
# from utils import *
from utillis_new_gpu import *
from scipy.interpolate import interp2d
from tqdm import tqdm
from fw_surf import *


def extract_values_from_nested(data):
    """
    递归函数，用于从嵌套的列表和字典中提取所有值，同时保留列表的维度。

    参数:
        data (list or dict): 嵌套的列表或字典结构。

    返回:
        list: 包含所有值的嵌套列表，保持原始维度。
    """
    if isinstance(data, dict):  # 如果当前数据是字典
        # 提取字典中的所有值，返回一个列表
        return [extract_values_from_nested(v) for v in data.values()]
    elif isinstance(data, list):  # 如果当前数据是列表
        # 递归处理列表中的每个元素，保持列表结构
        return [extract_values_from_nested(item) for item in data]
    else:
        # 如果是其他类型，直接返回该值
        return data


velocity_model = np.array([
    [10.0, 7.00, 3.50, 2.00],
    [10.0, 6.80, 3.40, 2.00],
    [10.0, 7.00, 3.50, 2.00],
    [10.0, 7.60, 3.80, 2.00],
    [10.0, 8.40, 4.20, 2.00],
    [10.0, 9.00, 4.50, 2.00],
    [10.0, 9.40, 4.70, 2.00],
    [10.0, 9.60, 4.80, 2.00],
    [10.0, 9.50, 4.75, 2.00],
])


def model_pre(obs, dd, model_num, ):
    f_list = obs[:, 0]
    v_list = obs[:, 1]
    t = 1 / f_list
    phase_vel_max = v_list.max() * 0.9
    tmax = t.max()
    dmax = tmax * v_list.max()


class grad_cal():
    def __init__(self,
                 inv_para,
                 target,
                 AK135_data=[],
                 device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
                 ):
        self.task_id = None
        self.inv_object = None
        self.beta = None
        self.rho = None
        self.alpha = None
        self.beta1 = None
        self.d = None
        self.init_model = None
        self.device = device

        self.t_list = 1/numpy2tensor(target[1].squeeze())
        self.c_list = numpy2tensor(target[0].squeeze())/1000

        self.lr = inv_para.lr
        self.es = inv_para.es
        self.vsrange = inv_para.vsrange
        self.initial_method = inv_para.inittal_method

        self.inv_para = inv_para
        self.target = target
        self.AK135_data = AK135_data

        # inversion result definition.to(self.device)
        self.inv_model = {
            "vs": [],
            "vp": [],
            "rho": [],
            "thick": []
        }
        self.forward_para = {
            'compress': True,
            'normalized': True,
            'compress_method': [],
        }

    def _layer_bounds(self, layer_definitions, model_num):
        vs_min = []
        vs_max = []
        d_min = []
        d_max = []
        for layer in layer_definitions:
            vs_min.append(float(layer["vs"][0]) / 1000.0)
            vs_max.append(float(layer["vs"][1]) / 1000.0)
            d_min.append(float(layer["d"][0]))
            d_max.append(float(layer["d"][1]))
        vs_min = torch.tensor(vs_min, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(model_num, 1)
        vs_max = torch.tensor(vs_max, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(model_num, 1)
        d_min = torch.tensor(d_min, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(model_num, 1)
        d_max = torch.tensor(d_max, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(model_num, 1)
        return vs_min, vs_max, d_min, d_max

    def _stabilize_initial_models(self, d, beta, rho, lower_b=None, upper_b=None):
        d_np = d.detach().cpu().numpy().copy()
        beta_np = beta.detach().cpu().numpy().copy()
        rho_np = rho.detach().cpu().numpy().copy()
        lower_np = lower_b.detach().cpu().numpy() if lower_b is not None else None
        upper_np = upper_b.detach().cpu().numpy() if upper_b is not None else None

        for i in range(beta_np.shape[0]):
            if beta_np.shape[1] > 2:
                beta_np[i, 1:-1] = 0.25 * beta_np[i, :-2] + 0.50 * beta_np[i, 1:-1] + 0.25 * beta_np[i, 2:]
                d_np[i, 1:-1] = 0.20 * d_np[i, :-2] + 0.60 * d_np[i, 1:-1] + 0.20 * d_np[i, 2:]
            for j in range(1, beta_np.shape[1]):
                prev = beta_np[i, j - 1]
                floor = prev * 0.90
                ceil = prev * 1.45
                beta_np[i, j] = np.clip(beta_np[i, j], floor, ceil)
                d_np[i, j] = max(d_np[i, j], d_np[i, j - 1] * 0.65)
            if lower_np is not None and upper_np is not None:
                beta_np[i] = np.clip(beta_np[i], lower_np[i], upper_np[i])
            rho_np[i] = np.clip(rho_np[i], 1.6, 3.0)

        d_t = torch.tensor(d_np, dtype=d.dtype, device=self.device)
        beta_t = torch.tensor(beta_np, dtype=beta.dtype, device=self.device)
        rho_t = torch.tensor(rho_np, dtype=rho.dtype, device=self.device)
        return d_t, beta_t, rho_t

    def _regularization(self, beta, d=None):
        eps = 1e-6
        rel_scale = torch.clamp(beta[:, :-1].abs(), min=eps)
        smooth_penalty = torch.mean(((beta[:, 1:] - beta[:, :-1]) / rel_scale) ** 2)
        max_rel_drop = 0.18 if self.inv_object == "vs" else 0.14
        trend_floor = beta[:, :-1] * (1.0 - max_rel_drop)
        trend_penalty = torch.mean(torch.relu(trend_floor - beta[:, 1:]) ** 2)
        if beta.shape[1] > 2:
            curvature_scale = torch.clamp(beta[:, 1:-1].abs(), min=eps)
            curvature_penalty = torch.mean(((beta[:, 2:] - 2 * beta[:, 1:-1] + beta[:, :-2]) / curvature_scale) ** 2)
        else:
            curvature_penalty = beta.new_tensor(0.0)
        reg = 0.018 * smooth_penalty + 0.018 * trend_penalty + 0.012 * curvature_penalty
        if d is not None and d.shape[1] > 1:
            d_scale = torch.clamp(d[:, :-1].abs(), min=1e-3)
            depth_smooth = torch.mean(((d[:, 1:] - d[:, :-1]) / d_scale) ** 2)
            reg = reg + 0.006 * depth_smooth
        return reg

    def _project_parameters(self, model, lower_b, upper_b, layer_mindepth=None, layer_maxdepth=None):
        with torch.no_grad():
            model.beta.data = torch.nan_to_num(model.beta.data, nan=0.0, posinf=0.0, neginf=0.0)
            model.beta.data.clip_(min=lower_b, max=upper_b)
            if model.beta.data.shape[1] > 1:
                max_rel_drop = 0.16 if self.inv_object == "vs" else 0.12
                rise_cap = 1.35 if self.inv_object == "vs" else 1.28
                for j in range(1, model.beta.data.shape[1]):
                    prev = model.beta.data[:, j - 1]
                    floor = prev * (1.0 - max_rel_drop)
                    ceil = torch.minimum(prev * rise_cap, upper_b[:, j])
                    model.beta.data[:, j] = torch.maximum(model.beta.data[:, j], floor)
                    model.beta.data[:, j] = torch.minimum(model.beta.data[:, j], ceil)
            if self.inv_object == "vsandd" and hasattr(model, "d"):
                model.d.data = torch.nan_to_num(model.d.data, nan=0.0, posinf=0.0, neginf=0.0)
                model.d.data.clip_(min=layer_mindepth, max=layer_maxdepth)
                if model.d.data.shape[1] > 1:
                    for j in range(1, model.d.data.shape[1]):
                        prev_d = model.d.data[:, j - 1]
                        floor_d = torch.maximum(layer_mindepth[:, j], prev_d * 0.60)
                        ceil_d = torch.minimum(layer_maxdepth[:, j], prev_d * 1.80)
                        model.d.data[:, j] = torch.maximum(model.d.data[:, j], floor_d)
                        model.d.data[:, j] = torch.minimum(model.d.data[:, j], ceil_d)

    def _sanitize_gradients(self, model):
        for para in model.parameters():
            if para.grad is None:
                continue
            para.grad.data = torch.nan_to_num(para.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

    def opt(self, calculator):
        if self.inv_para.optimizer == "Adam":
            return torch.optim.Adam(calculator.parameters(), lr=self.lr, betas=(0.9, 0.98))
        elif self.inv_para.optimizer == "AdamW":
            return torch.optim.AdamW(calculator.parameters(), lr=self.lr, weight_decay=1e-4, betas=(0.9, 0.98))
        elif self.inv_para.optimizer == "SGD":
            return torch.optim.SGD(calculator.parameters(), lr=self.lr, momentum=0.85, nesterov=True)
        else:
            raise NameError("The input optimizer {} can not find in Pytorch".format(self.inv_para.optimizer))

    def inv_pro(self, layer_definitions, callback_plotdata, callback_plotdata1, callback_plotdata2, task_id):
        # =========================
        # 1. Monte Carlo 初始化模型
        # =========================
        self.task_id = task_id

        models = monte_carlo_sampling(
            layer_params=layer_definitions,
            n_models=self.inv_para.model_num,
        )

        init_model = np.asarray(extract_values_from_nested(models))
        self.init_model = init_model

        # 物性参数（numpy → tensor）
        d = numpy2tensor(init_model[:, :, 0])
        beta = numpy2tensor(init_model[:, :, 1]) / 1000
        rho = numpy2tensor(init_model[:, :, 2])

        # 约定：alpha = beta
        alpha = beta

        # 保存原始参数
        self.d, self.beta, self.alpha, self.rho = d, beta, alpha, rho
        self.beta1 = beta  # 若仅作备份，可考虑删除

        # =========================
        # 2. device 转移
        # =========================
        d, alpha, beta, rho = [x.to(self.device) for x in (d, alpha, beta, rho)]

        c_list = torch.tile(self.c_list.to(self.device),
                            (self.inv_para.model_num, 1))
        t_list = torch.tile(self.t_list.to(self.device),
                            (self.inv_para.model_num, 1))

        # =========================
        # 3. Forward 模型 & 优化器
        # =========================
        self.inv_object = self.inv_para.inv_object

        model = FORWARD(
            c_list, d, alpha, beta, rho,
            self.forward_para,
            self.inv_object,
            self.initial_method,
            device=self.device
        ).to(self.device)

        optim = self.opt(model)

        iter_max = int(self.inv_para.iter_max)
        model_num = self.inv_para.model_num

        # =========================
        # 4. 反演边界条件
        # =========================
        lower_b, upper_b, layer_mindepth, layer_maxdepth = self._layer_bounds(layer_definitions, model_num)
        lower_b = torch.maximum(lower_b, (self.vsrange[0] * beta).to(self.device))
        upper_b = torch.maximum(upper_b, (self.vsrange[1] * beta).to(self.device))
        layer_mindepth = torch.maximum(layer_mindepth, torch.ones_like(layer_mindepth) * 1e-3)

        # =========================
        # 5. 结果缓存
        # =========================
        iter_vs = torch.zeros(
            iter_max, model_num, beta.shape[-1],
            device=self.device
        )

        iter_thick = torch.zeros(
            iter_max, model_num, d.shape[-1],
            device=self.device
        )

        loss_list = torch.full(
            (iter_max, model_num), 10.0,
            device=self.device
        )
        # iter_max = tqdm(range(iter_max))
        for index in range(iter_max):
            loss, DISP = model(c_list, t_list)
            optim.zero_grad()
            loss.backward(torch.ones_like(loss), retain_graph=True)
            if self.inv_object == "vsandd":
                for k, para in model.named_parameters():
                    # constrain the velocity
                    if k == 'beta':
                        para.data.clip_(min=lower_b, max=upper_b)
                        # last layer > second from the bottom
                        para.data[:,-1].clip_(min=(para.data[:,-2]).detach())
                        if self.device == "cpu":
                            iter_vs[index] = para.data.detach()
                        else:
                            iter_vs[index] = para.data.cpu().detach()
                    # constrain the depth
                    elif k == 'd':
                        with torch.no_grad():
                            para.data.clip_(min=layer_mindepth,
                                            max=layer_maxdepth)
                            # para.data[:, -1].clip_(min=(para.data[:, -2]).detach())
                            if self.device == "cpu":
                                iter_thick[index] = para.data.detach()
                            else:
                                iter_thick[index] = para.data.cpu().detach()
            else:
                for name, para in model.named_parameters():
                    if name == 'beta':
                        iter_thick[index] = d
                        para.grad.data = para.grad.data
                        if torch.isnan(para.data).any():
                            nan_mask = torch.isnan(para.data)
                            para.data[nan_mask] = self.beta1.to(self.device)[nan_mask]

                        para.data.clip_(min=lower_b, max=upper_b)
                        # last layer > second from the bottom
                        # para.data[0].clip_(min=lower_b)
                        # para.data[-1].clip_(min=(para.data[-2]).detach())
                        para.data[:, -1].clip_(min=(para.data[:, -2]).detach())
                        if self.device == "cpu":
                            iter_vs[index] = para.data.detach()
                        else:
                            iter_vs[index] = para.data.cpu().detach()
                    else:
                        continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optim.step()
            scheduler.step()
            loss_list[index] = loss.detach()
            min_loss = torch.min(loss_list[loss_list>0])
            # print('min_loss:')
            # print(min_loss)
            flat_index = torch.argmin(loss_list).item()
            # 将一维索引转换为二维坐标
            row = flat_index // loss_list.shape[1]  # 行坐标
            col = flat_index % loss_list.shape[1]  # 列坐标
            callback_plotdata2(np.array(iter_thick[row, col].cpu()), self.task_id)
            callback_plotdata(np.array(iter_vs[row, col].cpu()), self.task_id)
            if self.device == 'cpu':
                callback_plotdata1(self.task_id)
            else:
                callback_plotdata1(self.task_id)
            # if self.device == "cpu":
            #     iter_max.set_description(
            #         "Iter:{}\n"
            #         "lr:{}\n"
            #         "loss sum:{:.4}\n".format(index, scheduler.get_last_lr(), np.min(loss.detach().numpy())))
            # else:
            #     iter_max.set_description(
            #         "Iter:{}\n"
            #         "lr:{}\n"
            #         "loss sum:{:.4}\n".format(index, scheduler.get_last_lr(), np.min(loss.cpu().detach().numpy())))

            # callback
            # if (index > 0 and loss_list[index].min() - loss_list[index - 1].min() < self.es) or torch.isnan(loss_list[index].min()) and np.sum(loss.cpu().detach().numpy()) < 0.1:
            #     print('inv stopped \niter:%d \nloss:%f' % (index, loss_list[index].min()))
            #     break
        if self.device == "cpu":
            loss_list = list2numpy(loss_list)
            iter_vs = list2numpy(iter_vs)
            iter_thick = list2numpy(iter_thick)
        else:
            loss_list = loss_list.cpu().detach().numpy()
            iter_vs = iter_vs.cpu().detach().numpy()
            iter_thick = iter_thick.cpu().detach().numpy()

        best_num = np.argmin(loss_list, axis=0)
        if self.inv_object == "vsandd":
            inv_thick = iter_thick[best_num]
        else:
            inv_thick = list2numpy(self.init_model[:, 0])
        inv_vs = iter_vs[best_num]
        inv_vp = 0.9409 + 2.0947 * self.beta - 0.8206 * self.beta ** 2 + 0.2683 * self.beta ** 3 - 0.0251 * self.beta ** 4
        inv_rho = 1.6612 * self.alpha - 0.4721 * self.alpha ** 2 + 0.0671 * self.alpha ** 3 - 0.0043 * self.alpha ** 4 + 0.000106 * self.alpha ** 5
        self.inv_model = {
            "thick": inv_thick.tolist(),
            "vp": inv_vp.tolist(),
            "vs": inv_vs.tolist(),
            "rho": inv_rho.tolist()
        }
        self.inv_process = {
            "iter_vs": iter_vs.tolist(),
            "iter_thick": iter_thick.tolist(),
            "loss": loss_list.tolist()
        }

    def plot(self, example_ch):
        plt.figure()
        inv_res = self.inv_model
        inv_pro = self.inv_process
        loss = inv_pro['loss']
        v1 = inv_res['vs']
        v2 = inv_pro['iter_vs'][-1]

        dep = self.d
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
        plt.figure(figsize=(5, 10))
        for k in range(int(len(v1) / 20)):
            if k == 0:
                plt.step(v1[k * 20:(k + 1) * 20], dep[k * 20:(k + 1) * 20],
                         color='r', linewidth=3, zorder=2)
            plt.step(v1[k * 20:(k + 1) * 20], dep[k * 20:(k + 1) * 20], zorder=1)
        plt.gca().invert_yaxis()
        plt.title('Inv_result_%d' % example_ch)
        plt.xlabel('Vs(m/s)')
        plt.ylabel('Depth(m)')
        plt.show()
        plt.figure()
        plt.plot(loss, color='black')
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.title('Iteration_%d' % example_ch)
        plt.show()


def _grad_cal_inv_pro_stable(self, layer_definitions, callback_plotdata, callback_plotdata1, callback_plotdata2, task_id):
    self.task_id = task_id

    models = monte_carlo_sampling(
        layer_params=layer_definitions,
        n_models=self.inv_para.model_num,
    )
    init_model = np.asarray(extract_values_from_nested(models))
    self.init_model = init_model

    d = numpy2tensor(init_model[:, :, 0])
    beta = numpy2tensor(init_model[:, :, 1]) / 1000
    rho = numpy2tensor(init_model[:, :, 2])
    alpha = beta

    self.d, self.beta, self.alpha, self.rho = d, beta, alpha, rho
    self.beta1 = beta

    d, alpha, beta, rho = [x.to(self.device) for x in (d, alpha, beta, rho)]
    c_list = torch.tile(self.c_list.to(self.device), (self.inv_para.model_num, 1))
    t_list = torch.tile(self.t_list.to(self.device), (self.inv_para.model_num, 1))

    self.inv_object = self.inv_para.inv_object
    model = FORWARD(
        c_list, d, alpha, beta, rho,
        self.forward_para,
        self.inv_object,
        self.initial_method,
        device=self.device
    ).to(self.device)

    optim = self.opt(model)
    iter_max = int(self.inv_para.iter_max)
    model_num = self.inv_para.model_num
    lower_b, upper_b, layer_mindepth, layer_maxdepth = self._layer_bounds(layer_definitions, model_num)
    lower_b = torch.maximum(lower_b, (self.vsrange[0] * beta).to(self.device))
    upper_b = torch.maximum(upper_b, (self.vsrange[1] * beta).to(self.device))
    layer_mindepth = torch.maximum(layer_mindepth, torch.ones_like(layer_mindepth) * 1e-3)
    d, beta, rho = self._stabilize_initial_models(d, beta, rho, lower_b, upper_b)
    model.d.data.copy_(d)
    model.beta.data.copy_(beta)
    model.rho = rho
    self.d, self.beta, self.alpha, self.rho = d.detach().clone(), beta.detach().clone(), beta.detach().clone(), rho.detach().clone()

    iter_vs = torch.zeros(iter_max, model_num, beta.shape[-1], device=self.device)
    iter_thick = torch.zeros(iter_max, model_num, d.shape[-1], device=self.device)
    loss_list = torch.full((iter_max, model_num), 10.0, device=self.device)

    best_loss_val = float("inf")
    best_loss_vector = None
    best_vs = beta.detach().clone()
    best_thick = d.detach().clone()
    best_state = {
        "beta": beta.detach().clone(),
        "d": d.detach().clone(),
    }
    plateau_count = 0
    patience = max(12, int(getattr(self.inv_para, "step_size", 20)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=float(self.inv_para.gamma),
        patience=max(4, patience // 3),
        min_lr=max(self.lr * 0.05, 1e-4),
    )

    for index in range(iter_max):
        optim.zero_grad(set_to_none=True)
        data_loss, _ = model(c_list, t_list)
        robust_data_loss = torch.log1p(8.0 * torch.clamp(data_loss, min=0.0))
        reg = self._regularization(model.beta, model.d if self.inv_object == "vsandd" else None)
        total_loss_vec = robust_data_loss + reg
        total_loss = torch.mean(total_loss_vec)
        if not torch.isfinite(total_loss):
            with torch.no_grad():
                model.beta.data.copy_(best_state["beta"])
                if self.inv_object == "vsandd":
                    model.d.data.copy_(best_state["d"])
            break

        total_loss.backward()
        self._sanitize_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optim.step()
        self._project_parameters(model, lower_b, upper_b, layer_mindepth, layer_maxdepth)

        current_vs = model.beta.detach().clone()
        current_thick = model.d.detach().clone() if self.inv_object == "vsandd" else d.detach().clone()
        iter_vs[index] = current_vs
        iter_thick[index] = current_thick
        loss_list[index] = total_loss_vec.detach()

        current_best = float(torch.min(total_loss_vec).detach().cpu())
        scheduler.step(current_best)
        if current_best + float(self.es) < best_loss_val:
            best_loss_val = current_best
            best_loss_vector = total_loss_vec.detach().clone()
            best_vs = current_vs.clone()
            best_thick = current_thick.clone()
            best_state = {
                "beta": model.beta.detach().clone(),
                "d": current_thick.clone(),
            }
            plateau_count = 0
        else:
            plateau_count += 1

        flat_index = torch.argmin(loss_list[:index + 1]).item()
        row = flat_index // loss_list.shape[1]
        col = flat_index % loss_list.shape[1]
        callback_plotdata2(np.array(iter_thick[row, col].cpu()), self.task_id)
        callback_plotdata(np.array(iter_vs[row, col].cpu()), self.task_id)
        callback_plotdata1(self.task_id)

        if plateau_count >= patience:
            break

    if self.device == "cpu":
        loss_list = list2numpy(loss_list)
        iter_vs = list2numpy(iter_vs)
        iter_thick = list2numpy(iter_thick)
    else:
        loss_list = loss_list.cpu().detach().numpy()
        iter_vs = iter_vs.cpu().detach().numpy()
        iter_thick = iter_thick.cpu().detach().numpy()

    final_vs_tensor = best_vs if best_vs is not None else model.beta.detach().clone()
    final_thick_tensor = best_thick if best_thick is not None else (model.d.detach().clone() if self.inv_object == "vsandd" else d.detach().clone())
    inv_vs = final_vs_tensor.cpu().detach().numpy()
    inv_thick = final_thick_tensor.cpu().detach().numpy()
    inv_vp = 0.9409 + 2.0947 * final_vs_tensor - 0.8206 * final_vs_tensor ** 2 + 0.2683 * final_vs_tensor ** 3 - 0.0251 * final_vs_tensor ** 4
    inv_rho = 1.6612 * inv_vp - 0.4721 * inv_vp ** 2 + 0.0671 * inv_vp ** 3 - 0.0043 * inv_vp ** 4 + 0.000106 * inv_vp ** 5
    self.inv_model = {
        "thick": np.asarray(inv_thick).tolist(),
        "vp": inv_vp.cpu().detach().numpy().tolist(),
        "vs": inv_vs.tolist(),
        "rho": inv_rho.cpu().detach().numpy().tolist()
    }
    self.inv_process = {
        "iter_vs": iter_vs.tolist(),
        "iter_thick": iter_thick.tolist(),
        "loss": loss_list.tolist(),
        "best_loss": best_loss_vector.cpu().detach().numpy().tolist() if best_loss_vector is not None else []
    }


grad_cal.inv_pro = _grad_cal_inv_pro_stable

def Forward_t(self):
    alpha = numpy2tensor(list2numpy(velocity_model[:, 1])).to('cpu')
    d = velocity_model[:, 0]
    beta = velocity_model[:, 2]
    rho = velocity_model[:, 3]
    t = np.logspace(0.0, 3.0, 100)
    result = [[t[i] for i in range(100)] for num in range(100, 1001)]
    ttt = np.array(result)
    result = [[i for i in range(100, 1001)] for num in range(1, 101)]
    tt = np.array(result)
    ttt = ttt.reshape(100 * 901)
    ttt.sort()
    tt = tt / 100
    tt = tt / 2
    tt = tt / 2 + 3
    tt = tt.reshape(90100)
    d = np.expand_dims(d, axis=0)
    rho = np.expand_dims(rho, axis=0)
    tt = np.expand_dims(tt, axis=0)
    ttt = np.expand_dims(ttt, axis=0)
    A = dltar4_vector(tt, ttt, d, alpha, beta, rho, llw=0, device="cpu")
    A1 = A.reshape(100, 901)
    A1 = np.array(A1)
    A1[A1 < 0] = -1
    A1[A1 > 0] = 1
    vel_tick = np.arange(A1.shape[1]) * 0.0025 + 2.75
    Period_tick = np.arange(A1.shape[0])
    Period_tick_int = np.linspace(0, Period_tick[-1] - 0.01, 1801)
    vel_tick_int = np.linspace(vel_tick[0], vel_tick[-1] - 0.01, 1801)
    func = interp2d(Period_tick, vel_tick, A1.transpose(), kind='cubic')
    A11 = func(Period_tick_int, vel_tick_int)
    plt.pcolormesh(Period_tick_int, vel_tick_int, A11, cmap='jet')
    plt.ylabel('Phase Vel(km/s)')
    plt.xlabel('Period(s)')
    plt.title('DispCurve')
    plt.show()
