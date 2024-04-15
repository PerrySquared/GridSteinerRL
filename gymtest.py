import gymnasium as gym
import gym_examples
import numpy as np
import time

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C


import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



def make_env(env_id, id):
    def _init():

        env = gym.make(env_id, render_mode="human")
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset()
        return env

    return _init

if __name__ == "__main__":
    
    PROCESSES_TO_TEST = 4
    NUM_EXPERIMENTS = 3  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 1000000
    # Number of episodes for evaluation
    EVAL_EPS = 20
    ALGO = A2C

    # We will create one environment to evaluate the agent on
    eval_env = gym.make('gym_examples/GridWorld-v0')

    reward_averages = []
    reward_std = []
    training_times = []
    total_procs = PROCESSES_TO_TEST

    print(f"Running for n_procs = {total_procs}")
    if total_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        train_env = DummyVecEnv([lambda: gym.make('gym_examples/GridWorld-v0')])
    else:
        # Here we use the "fork" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env('gym_examples/GridWorld-v0', n_envs=total_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
        train_env = SubprocVecEnv(
            [make_env('gym_examples/GridWorld-v0', i) for i in range(total_procs)],
            start_method="spawn",
        )

    rewards = []
    times = []

    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        train_env.reset()
        model = ALGO("MultiInputPolicy", train_env, verbose=0, device="cpu")
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS)
        times.append(time.time() - start)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)
    # Important: when using subprocesses, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))




# if __name__ == "__main__":
    

    # env = gym.vector.AsyncVectorEnv([
    #     lambda: gym.make('gym_examples/GridWorld-v0'),
    #     lambda: gym.make('gym_examples/GridWorld-v0'),
    #     lambda: gym.make('gym_examples/GridWorld-v0'),
    #     lambda: gym.make('gym_examples/GridWorld-v0'),
    #     lambda: gym.make('gym_examples/GridWorld-v0'),
    #     lambda: gym.make('gym_examples/GridWorld-v0'),
    #     lambda: gym.make('gym_examples/GridWorld-v0'),
    #     lambda: gym.make('gym_examples/GridWorld-v0')

    # ])

    # # env = gym.make('gym_examples/GridWorld-v0')    

    # model = A2C("MultiInputPolicy", env, verbose=1)
    # model.learn(total_timesteps=1e07)
    # model.save("grid")

    # del model # remove to demonstrate saving and loading

    # model = A2C.load("grid")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, truncs, info = env.step(action)
    #     env.render("human")