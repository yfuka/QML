import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMultiplyLayer(nn.Module):

    def __init__(self, in_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomMultiplyLayer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty((in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mul(input, self.weight)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)