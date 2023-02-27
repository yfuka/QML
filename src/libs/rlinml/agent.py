# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Pytorchの公式チュートリアルをもとにしたコードになります。

import os
import sys
#Topディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from src.libs.custommodel import hybridmodel, classicalmodel

from collections import namedtuple, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Subclass of namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, env, hyper_params, model_type: str, device, dtype=torch.float32):
        # 課題の状態と行動の数を設定
        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        self.device = device
        self.dtype = dtype
        self.model_type = model_type

        if self.model_type == "hybrid":
            self.c_depth, self.backend, self.shots = hyper_params['c_depth'], hyper_params['backend'], hyper_params['shots']
            self.policy_net = hybridmodel.QQN(self.n_observations, self.n_actions, self.c_depth, self.backend, \
                                                self.shots, self.device, self.dtype).to(self.device)

            # https://pytorch.org/docs/stable/optim.html
            LRS = [1e-3, 1e-3, 1e-1]
            self.optimizer = optim.AdamW([{'params':param, 'lr':lr} for param, lr in zip(self.policy_net.parameters(), LRS)], amsgrad=True)

            self.target_net = hybridmodel.QQN(self.n_observations, self.n_actions, self.c_depth, self.backend, \
                                                self.shots, self.device, self.dtype).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict()) # targetネットワークと同期

        elif self.model_type == "classical":
            self.policy_net = classicalmodel.DQN(self.n_observations, self.n_actions).to(self.device)
            LR = hyper_params['LR']
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
            self.target_net = classicalmodel.DQN(self.n_observations, self.n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict()) # targetネットワークと同期
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        self.GAMMA, self.BATCH_SIZE, self.memory = hyper_params['GAMMA'], hyper_params['BATCH_SIZE'], ReplayMemory(hyper_params['MEMORY_SIZE'])

        print("The initial parameters of the nets:")
        print(self.policy_net.state_dict())
        
    def get_action(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was found,
                # so we pick action with the larger expected reward.
                state = torch.tensor(state, dtype=self.dtype, device=self.device).unsqueeze(0)
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long) # action_spaceを選択するため int/long

    def store_experience(self, state, action, next_state, reward):
        state = torch.tensor(state, dtype=self.dtype, device=self.device).unsqueeze(0)
        if next_state is not None:
            next_state = torch.tensor(next_state, dtype=self.dtype, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=self.dtype, device=self.device)
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.model_type == "classical":
        # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def softupdate_target_network(self, TAU):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, path=os.getcwd()):
        # Save the parameters of the model
        torch.save(self.policy_net.state_dict(), path + "\model_" + self.model_type + ".bin")