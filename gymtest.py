import gymnasium as gym
import gym_examples
import numpy as np
import time
import matplotlib.pyplot as plt


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, A2C
import sys
np.set_printoptions(threshold=sys.maxsize)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def make_env(env_id, id):
    def _init():

        env = gym.make(env_id, render_mode=None)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset()
        return env

    return _init

if __name__ == "__main__":
    
    PROCESSES_TO_TEST = 4
    NUM_EXPERIMENTS = 1  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 2000000
    EVAL_EPS = 100
    ALGO = PPO

    # We will create one environment to evaluate the agent on
    eval_env = gym.make('gym_examples/GridWorld-v0', render_mode=None)

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
        model = ALGO(
                        "MultiInputPolicy", 
                        train_env,
                        verbose=1, 
                        device="cuda",
                        learning_rate=7e-5, 
                        batch_size=64,
                        gamma=0.95, 
                        ent_coef=0.05, 
                        clip_range=0.15,
                        tensorboard_log="./gp_tensorboard/",
                    )
        print(f"Using device: {model.device}")

        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True, callback=EvalCallback(train_env, best_model_save_path="./best_models", n_eval_episodes=EVAL_EPS, eval_freq=2048, verbose=1))
        
        print("\n=============\nEVAL STARTED\n=============\n")
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)
        
    # Important: when using subprocesses, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    
    model.save("ppo_model")
    print("\nMODEL SAVED\n")
    
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))



    def plot_training_results(training_steps_per_second, reward_averages, reward_std):
        """
        Utility function for plotting the results of training

        :param training_steps_per_second: List[double]
        :param reward_averages: List[double]
        :param reward_std: List[double]
        """
        plt.figure(figsize=(9, 4))
        plt.subplots_adjust(wspace=0.5)
        plt.subplot(1, 2, 1)
        plt.errorbar(
            PROCESSES_TO_TEST,
            reward_averages,
            yerr=reward_std,
            capsize=2,
            c="k",
            marker="o",
        )
        plt.xlabel("Process")
        plt.ylabel("Average return")
        plt.subplot(1, 2, 2)
        plt.bar(range(PROCESSES_TO_TEST), training_steps_per_second)
        plt.xlabel("Process")
        plt.ylabel("Training steps per second")

        plt.show()
        
    # training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

    # plot_training_results(training_steps_per_second, reward_averages, reward_std)
    
    

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