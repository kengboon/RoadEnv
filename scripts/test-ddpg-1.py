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
    "duration": 60,
    "obstacle_preset": 1
})

# %% Get dimensions
print('Observation space', env.observation_space.shape)
print('Action shape', env.action_space.shape)

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
print("State dim:", state_dim, "Action dim:", action_dim, "Max action:", max_action)

# %% Create DRL agent
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from rl_algorithms import DDPGAgent
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    hidden_size=128)
agent.load('models/ddpg-230620090010/0', device)

obs, info = env.reset()
done = truncated = False

num_episode = 20
for episode in range(num_episode):
    print('Episode', episode+1)
    num_steps = 0
    episode_reward = 0

    while True: # Use config["duration"] to truncate
        action = agent.get_action(obs)
        print(action)
        obs, reward, done, truncated, info = env.step(action)
        num_steps += 1
        episode_reward += reward
        env.render() # Note: Do not render during training

        if done or truncated:
            print(env.get_performance())
            obs, info = env.reset()
            break

    episode_log = {
        "Episode": episode+1,
        "Time steps": num_steps,
        "Episode Rewards": episode_reward,
        "Average Rewards": episode_reward / num_steps,
    }
    print(episode_log)

env.close()