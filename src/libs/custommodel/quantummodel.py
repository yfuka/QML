import os
import sys
#Topディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.libs.customqc import reup_qc, cal_expval

import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

class ReUploadingPQCLayer(nn.Module):
    
    def __init__(self, n_observations: int, n_actions: int, c_depth: int, backend, shots: int, device=None, dtype=torch.float32):
        super(ReUploadingPQCLayer, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_qubits = n_observations
        self.c_depth  = c_depth
        self.backend = backend
        self.shots = shots
        self.device = device
        self.dtype = dtype
        self.re_uploading_PQC = reup_qc.ReUploadingPQC(n_qubits=self.n_qubits, c_depth=self.c_depth, backend=self.backend, shots=self.shots)
        self.thetas = nn.Parameter(torch.rand(self.re_uploading_PQC.num_parameters, dtype=self.dtype, device=self.device) * np.pi)
        
    def forward(self, xs: List[float]):
        def f(x: torch.Tensor, thetas: torch.Tensor) -> np.ndarray:
            input_x = x.tolist() * self.c_depth # re-uploadingの回数だけxを複製
            result = self.re_uploading_PQC.run(input_x, thetas.tolist())
            expectation_values = cal_expval.cal_expectation_values(result)
            expectation_values_scaled = (1 + expectation_values) /2 # 0 ~ 1に規格化
            return expectation_values_scaled.tolist()
        def f_each(xs: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
            return torch.tensor([f(x, thetas) for x in xs], dtype=self.dtype, device=self.device)
        return DifferenceByParameterShift.apply(f_each, self.device, xs, self.thetas)

# https://qiita.com/gyu-don/items/dbbe0f87bff6553655b0
# を参考に
class DifferenceByParameterShift(Function):
    """
    The gradients of a runnable quantum circuit layer can be obtained following Parameter-Shift rule, not the auto grad of Pytorch.
    """

    @staticmethod
    def forward(ctx, f_each, device, xs, thetas):
        ys = f_each(xs, thetas)
        ctx.save_for_backward(xs, ys, thetas)
        ctx.f_each = f_each
        ctx.device = device
        return ys

# https://qiita.com/gyu-don/items/ace6f69c07feb92d49b1#f%E3%82%92f_str%E3%81%AB%E5%A4%89%E3%81%88%E3%81%A6%E3%81%BF%E3%82%8B
# https://qiita.com/notori48/items/0ab84113afc9204d3e90
# https://www.investor-daiki.com/it/qiskit-parameter-shift
    @staticmethod
    def backward(ctx, grad_output):
        xs, ys, thetas = ctx.saved_tensors

        dtheta = np.pi/2
        diff_thetas = []
        thetas = thetas.detach() # auto_gradを使わず
        for i in range(len(thetas)):
            thetas[i] += dtheta
            forward = ctx.f_each(xs, thetas)
            thetas[i] -= 2 * dtheta
            backward = ctx.f_each(xs, thetas)
            gradient = 0.5 * (forward - backward)
            diff_thetas.append(torch.sum(grad_output * gradient))
            thetas[i] += dtheta # shift前に戻す
        diff_thetas = torch.tensor(diff_thetas, dtype=torch.float32, device=ctx.device)

        dx = np.pi/2
        diff_xs_list = []
        xs = xs.detach() # auto_gradを使わず
        for i in range(xs.size()[1]):
            dxtensor = torch.zeros(xs.size()[1])
            dxtensor[i] = dx
            dxtensor = dxtensor.repeat(xs.size()[0], 1)

            xs += dxtensor
            forward = ctx.f_each(xs, thetas)
            xs -= 2 * dxtensor
            backward = ctx.f_each(xs, thetas)
            gradient = 0.5 * (forward - backward)
            diff_xs_list.append(torch.sum(grad_output * gradient, 1, keepdim=True)) # 3 * 2 -> 3 * 1
            xs += dxtensor # shift前に戻す

        diff_xs = torch.cat(diff_xs_list, dim=1)
        diff_xs.to(device=ctx.device, dtype=torch.float32)

        return None, None, diff_xs, diff_thetas

