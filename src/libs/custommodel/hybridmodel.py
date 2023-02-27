import os
import sys
#Topディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import torch.nn as nn

from src.libs.custommodel import classicalmodel, quantummodel

class QQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int, c_depth: int, backend, shots: int, device=None, dtype=torch.float32):
        super(QQN, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_qubits = n_observations
        self.c_depth  = c_depth
        self.backend = backend
        self.shots = shots
        self.device = device
        self.dtype = dtype
        self.custom_multiply_layer1 = classicalmodel.CustomMultiplyLayer(self.n_observations)
        self.re_uploading_PQC_layer = quantummodel.ReUploadingPQCLayer(self.n_observations, self.n_actions, self.c_depth, self.backend, \
                                                                        self.shots, self.device, self.dtype)
        self.custom_multiply_layer2 = classicalmodel.CustomMultiplyLayer(self.n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        weighted_x = self.custom_multiply_layer1(x)
        angle = torch.arctan(weighted_x)
        Q = self.re_uploading_PQC_layer(angle)
        weighted_Q = self.custom_multiply_layer2(Q)
        return weighted_Q