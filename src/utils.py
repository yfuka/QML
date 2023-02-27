from typing import List
import matplotlib.pyplot as plt
import os
import gym
import torch
import ast

def prepare_env(env_name: str = 'CartPole-v1'):
    if gym.__version__[:4] == '0.26':
        env = gym.make(env_name)
    elif gym.__version__[:4] == '0.25':
        env = gym.make(env_name, new_step_api=True)
    else:
        raise ImportError(f"Requires gym v25 or v26, actual version: {gym.__version__}")
    return env


def prepare_device():
    # prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "cuda"
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    return device, device_name

def save_reward_change(reward_change: List, path: str = os.getcwd(), file_name:str = "reward_change.txt"):
    reward_change_str = str(reward_change)

    with open(path + "\\" + file_name, "w") as file:
        file.write(reward_change_str)

def load_reward_change(path: str = os.getcwd(), file_name:str = "reward_change.txt"):

    with open(path + "\\" + file_name, "r") as file:
        contents = file.readlines()
        reward_change = ast.literal_eval(contents[0])

    return reward_change

def plot_reward_change(reward_change: List):
    plt.figure(figsize=(10,5))
    plt.plot(reward_change)
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.show()