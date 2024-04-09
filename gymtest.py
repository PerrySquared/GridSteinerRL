import gym
import gym_examples

env = gym.make('gym_examples/GridWorld-v0')
obs, info = env.reset()

while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    if done == True:
        env.reset()