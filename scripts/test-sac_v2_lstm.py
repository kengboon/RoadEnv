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
    "duration": 999,
    "obstacle_preset": None
})
#env.config["obstacle_count"] = 0
#env.config["pedestrians"]["count"] = 0

# %% Get dimensions
print('Observation space', env.observation_space.shape)
print('Action shape', env.action_space.shape)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
print("State dim:", state_dim, "Action dim:", action_dim, "Max action:", max_action)

# %% Make DRL Trainer
hidden_dim = 512

from rl_algorithms2.sac_v2_lstm import SAC_Trainer, replay_buffer
trainer = SAC_Trainer(
    state_space=env.observation_space,
    action_space=env.action_space,
    action_range=max_action,
    hidden_dim=hidden_dim,
    replay_buffer=replay_buffer,
)

model_type = "sac_v2_lstm"
train_id = "230703210825"
episode = "7500"
model_dir = "../../data/models/" + model_type + "-" + train_id + "/" + episode
trainer.load_model(model_dir)

# %% Testing
obs, info = env.reset()
done = truncated = False
num_episode = 20

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
last_action = None
hidden_out = None
def random_action():
    global last_action, hidden_out
    last_action = env.action_space.sample()
    # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
    hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), \
        torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))
    
for episode in range(num_episode):
    print('Episode', episode+1)

    num_steps = 0
    episode_reward = 0

    random_action()

    while True: # Use config["duration"] to truncate
        hidden_in = hidden_out
        action, hidden_out = trainer.policy_net.get_action(obs, last_action, hidden_in, deterministic=True)
        #print(action)

        next_obs, reward, done, truncated, info = env.step(action)
        last_action = action

        print(env.vehicle.velocity[0] * 3.6)

        obs = next_obs
        num_steps += 1
        episode_reward += reward
        env.render() # Note: Do not render during training

        if done or truncated:
            env_perf = env.get_performance()
            obs, info = env.reset()
            break

    episode_log = {
        "Episode": episode+1,
        "Time steps": num_steps,
        "Episode Rewards": episode_reward,
        "Average Rewards": episode_reward / num_steps,
    }
    episode_log.update(env_perf)
    print(episode_log)
env.close()