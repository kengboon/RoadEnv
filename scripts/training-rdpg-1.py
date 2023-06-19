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
from datetime import datetime
train_id = datetime.now().strftime("%y%m%d%H%M%S")
model_dir = "models/rdpg-" + train_id

from rl_algorithms import DDPGAgent
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    hidden_size=128,
    recurrent=True)

# %% Training
training_logs = []

obs, info = env.reset()
done = truncated = False

import numpy as np
max_epsilon = 1
min_epsilon = 0.05
decay_rate = 0.0005
num_episode = 50000
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

    episode_log = {
        "Episode": episode+1,
        "Time steps": num_steps,
        "Episode Rewards": episode_reward,
        "Average Rewards": episode_reward / num_steps,
        "Epsilon": epsilon,
    }
    episode_log.update(agent.get_log())
    print(episode_log)
    training_logs.append(episode_log)
agent.save(model_dir)

import csv, os
os.makedirs("logs", exist_ok=True)
train_log_path = "logs/rdpg-" + train_id + ".csv"
with open(train_log_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=episode_log.keys())
    writer.writeheader()
    writer.writerows(training_logs)
print("Log:", train_log_path)

env.close()