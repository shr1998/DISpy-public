# -*- coding: utf-8 -*-
# @Time : 2023/12/23 14:56
# @Site :
# @File : main.py
# @Software: PyCharm
# from arg_get import gen_init_model as gen1
import numpy as np

import _surf96_vectorAll_gpu as surf96_vector_all_gpu
import torch

ifunc_list = {
    "dunkin": {"love": 1, "rayleigh": 2},
    "fast-delta": {"love": 1, "rayleigh": 3},
}


class FORWARD(torch.nn.Module):
    def __init__(self, Clist, d, a, b, rho, forward_para,
                 inv_object, initial_method, device="cpu"):
        super(FORWARD, self).__init__()
        self.llw = None
        self.d = d
        if inv_object == "vsandd":
            self.d = torch.nn.Parameter(self.d)
        wave = "rayleigh"
        algorithm = "dunkin"
        self.a = a
        self.rho = rho
        self.beta = torch.nn.Parameter(b)
        self.object = inv_object
        self.ifunc = ifunc_list[algorithm][wave]
        self.Clist = Clist
        self.device = device
        self.initial_method = initial_method
        self.para = forward_para
        # self.linear_layer = torch.nn.Linear(int(self.Clist.shape[-1]), int(self.Clist.shape[-1] / 1000))
        # self.gen = gen1()
        # self.gen.to(device)
        self.len = len

    def forward(self, c_list, t_list):
        # self.d = d
        # ref_model = torch.cat((d, a, b, rho), dim=0)
        # models = self.gen(ref_model)
        # models = torch.permute(models, (2, 0, 1))
        # self.b = models[:, 2, :]
        # self.d = models[:, 0, :]
        # self.alpha = models[:, 1, :]
        # self.rho = models[:, 3, :]
        # c_list = c_list.repeat(self.len(self.b), 1)
        # t_list = t_list.repeat(self.len(self.b), 1)

        self.llw = -1
        if self.initial_method == "Brocher":
            self.alpha = 0.9409 + 2.0947 * self.beta - 0.8206 * self.beta ** 2 + 0.2683 * self.beta ** 3 - 0.0251 * self.beta ** 4
            self.rho = 1.6612 * self.alpha - 0.4721 * self.alpha ** 2 + 0.0671 * self.alpha ** 3 - 0.0043 * self.alpha ** 4 + 0.000106 * self.alpha ** 5
        else:
            self.alpha = 1.16 * self.beta + 1.36
            self.rho = 0.541 + 0.360 * self.alpha - 0.109 * self.alpha ** 2 + 0.005 * self.alpha ** 3
        # F = surf96_vecter_single_gpu.dltar_vector(self.c_list, self.t_list, self.d, self.a, self.b, self.rho, self.ifunc, self.llw,
        #                                       device=self.device)
        F = surf96_vector_all_gpu.dltar_vector(c_list, t_list, self.d, self.alpha, self.beta, self.rho, self.ifunc,
                                               self.llw, device=self.device)
        temp_F = F
        if self.para['compress']:
            if self.para['normalized']:
                # compress with the normalized result
                # with torch.no_grad():
                #     Olist = self.t_list.reshape(1,-1)
                # det = surf_matrix_iter_gpu.dltar_matrix(self.Clist, Olist, self.d, self.a, self.b, self.rho, self.ifunc, self.llw, device=self.device)
                amp = torch.max(F, dim=1, keepdim=True).values - torch.min(F, dim=1, keepdim=True).values
                F = F / torch.clamp(amp, min=1e-6)
                # F = F / (torch.max(F, dim=1).values - torch.min(F, dim=1).values)
                if self.para['compress_method'] == "log":
                    F = torch.log(torch.abs(F) + 1)
                elif self.para['compress_method'] == "exp":
                    # det = det/(torch.max(det,dim=0).values - torch.min(det,dim=0).values)
                    F = 1e-1 ** torch.abs(F) - 1
                    # det = (1e-1)**torch.abs(det) - 1
            else:
                F = 1e-1 ** torch.abs(F) - 1
        # F = self.linear_layer(F)
        F_all = torch.mean(torch.abs(F), dim=1)
        return F_all, temp_F
