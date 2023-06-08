# %% Working directory
import sys, os
print('Working dir:', os.getcwd())
# In order to import modules from directories
if os.path.exists('road_env'):
    sys.path.append('.')
else:
    sys.path.append('..')

# %% Import
import gymnasium as gym
import matplotlib.pyplot as plt

# %% Register environment
from road_env import register_road_envs
register_road_envs()

# %% Print EnvSpec
#print(gym.spec('urban-road-v0'))

# %% Make environment
env = gym.make('urban-road-v0', render_mode='rgb_array')

print('# Action space ', env.action_space.shape)
print(env.action_space.sample())

print('# Observation space', env.observation_space.shape)
print(env.observation_space.sample())

env.configure({
    "random_seed": 10,
    "duration": 60, # Maximum duration (s) per episode
})
env.reset()

# %% Show display
plt.imshow(env.render())

# %% Execution
num_episodes = 20
for episode in range(num_episodes):
    print('Episode', episode+1)
    num_steps = 0
    episode_reward = 0

    while True: # Use config["duration"] to truncate
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        num_steps += 1
        env.render() # Note: Do not render during training

        episode_reward += reward

        if done or truncated:
            obs, info = env.reset()
            break

    print('Total steps:', num_steps, ', Total reward:', episode_reward)

env.close()