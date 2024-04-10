
import gym 
import gym_examples 
from stable_baselines3 import A2C 
 
env = gym.make('gym_examples/GridWorld-v0') 
         
 
model = A2C("MultiInputPolicy", env, verbose=1) 
model.learn(total_timesteps=1e07) 
model.save("grid") 
 
del model # remove to demonstrate saving and loading 
 
model = A2C.load("grid") 
 
obs = env.reset() 
while True: 
    action, _states = model.predict(obs) 
    obs, rewards, dones, info = env.step(action) 
    env.render("human")