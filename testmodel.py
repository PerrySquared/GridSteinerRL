
import gymnasium as gym 
import gym_examples 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import time
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3 import PPO 
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.evaluation import evaluate_policy
from gym_examples.envs.grid_world import TEMP_GENERAL_OVERFLOW
 
# Parallel environments 
vec_env = make_vec_env("gym_examples/GridWorld-v0") 
 
model = PPO.load("./final_models/ppo_model_5pin.zip", env=vec_env)

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
env_counter = 0

print("clock started")
t0 = time.time()

while env_counter < 1000:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if(dones):
        env_counter += 1
    episode_starts = dones
    
t1 = time.time()

print("clock finished")
print(t1-t0)

plt.imshow(TEMP_GENERAL_OVERFLOW, cmap='hot', interpolation='nearest')
plt.show()
print(TEMP_GENERAL_OVERFLOW.sum()) 