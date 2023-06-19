# %% Working directory
import sys, os
if os.path.exists("road_env"):
    sys.path.append(".")
else:
    sys.path.append("..")

# %% Register environment
from road_env import register_road_envs
register_road_envs()

# %% Make environment
import gymnasium as gym
env = gym.make("urban-road-v0", render_mode="rgb_array")
env.configure({
    "random_seed": None,
    "duration": 60
})

# %% Get dimensions
print('Observation space', env.observation_space.shape)
print('Action shape', env.action_space.shape)

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
print("State dim:", state_dim, "Action dim:", action_dim, "Max action:", max_action)

# %% Create DRL agent
from rl_algorithms import DDPGAgent
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    hidden_size=128)

# %% Training
obs, info = env.reset()
done = truncated = False
total_reward = 0

import numpy as np
max_epsilon = 1.
min_epsilon = 0.05
decay_rate = 0.0005
num_episode = 20
for episode in range(num_episode):
    print('Episode', episode+1)
    num_steps = 0
    episode_reward = 0
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    while True: # Use config["duration"] to truncate
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)
        epsilon -= 0.01
        next_obs, reward, done, truncated, info = env.step(action)

        # Update agent
        agent.add_to_replay_buffer(obs, action, reward, next_obs, done)
        agent.update()

        obs = next_obs
        num_steps += 1
        episode_reward += reward
        #env.render() # Note: Do not render during training

        if done or truncated:
            obs, info = env.reset()
            break

    print('Total steps:', num_steps, ', Total reward:', episode_reward)

env.close()
