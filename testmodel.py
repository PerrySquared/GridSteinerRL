
import gymnasium as gym 
import gym_examples 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3 import PPO 
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.evaluation import evaluate_policy 
 
# Parallel environments 
vec_env = make_vec_env("gym_examples/GridWorld-v0") 
 
model = PPO.load("./best_models/best_model.zip", env=vec_env)

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones