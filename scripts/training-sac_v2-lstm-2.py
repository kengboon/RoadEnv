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
    "duration": 100,
    "obstacle_preset": 4
})

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

from datetime import datetime
model_type = "sac_v2_lstm"
train_id = datetime.now().strftime("%y%m%d%H%M%S")
model_dir = "models/" + model_type + "-" + train_id

# %% Training
import csv, os
training_logs = []
def save_training_logs(writeheader=False):
    os.makedirs("logs/train", exist_ok=True)
    train_log_path = "logs/train/" + model_type + "-" + train_id + ".csv"
    with open(train_log_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=episode_log.keys())
        if writeheader:
            writer.writeheader()
        writer.writerows(training_logs)
    training_logs.clear()

time_step_rewards = []
def save_avg_rewards():
    os.makedirs("logs/train", exist_ok=True)
    log_path = "logs/train/" + model_type + "-" + train_id + "-rewards.csv"
    with open(log_path, "a", newline="") as file:
        file.write(str(sum(time_step_rewards) / len(time_step_rewards)))
        file.write("\n")
    time_step_rewards.clear()

obs, info = env.reset()
done = truncated = False

import time
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
AUTO_ENTROPY=True
DETERMINISTIC = False
max_epsilon = 0
min_epsilon = 0
decay_rate = 0.0005
num_episode = 10000
save_interval = 100
update_interval_steps = 32
last_update_step = 99999
batch_size = 2 # Episode
size_per_batch = 32 #
update_itr = 1
total_start_time = time.time()

episode_state = []
episode_action = []
episode_last_action = []
episode_rewards = []
episode_next_state = []
episode_done = []
for episode in range(num_episode):
    print('Episode', episode+1)
    obs, info = env.reset()

    start_time = time.time()
    num_steps = 0
    episode_reward = 0
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    last_action = env.action_space.sample()
    # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
    hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), \
        torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))

    while True: # Use config["duration"] to truncate
        hidden_in = hidden_out
        action, hidden_out = trainer.policy_net.get_action(obs, last_action, hidden_in, deterministic=DETERMINISTIC)
        #print(action)
        next_obs, reward, done, truncated, info = env.step(action)

        if num_steps == 0:
            # Initialize hidden state
            ini_hidden_in = hidden_in
            ini_hidden_out = hidden_out

        episode_state.append(obs)
        episode_action.append(action)
        episode_last_action.append(last_action)
        episode_rewards.append(reward)
        episode_next_state.append(next_obs)
        episode_done.append(done)

        if len(episode_state) == size_per_batch:
            # Push to replay buffer
            replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_rewards, episode_next_state, episode_done)
            # Reset - note: point to new empty list
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_rewards = []
            episode_next_state = []
            episode_done = []

        # Update agent
        if len(replay_buffer) > batch_size and last_update_step >= update_interval_steps:
            for i in range(update_itr):
                print("Update SAC-LSTM...")
                trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)
            last_update_step = 0
        else:
            last_update_step += 1

        obs = next_obs
        last_action = action
        num_steps += 1
        episode_reward += reward
        #env.render() # Note: Do not render during training

        if done or truncated:
            env_perf = env.get_performance()
            break

    end_time = time.time()

    episode_log = {
        "Episode": episode+1,
        "Time steps": num_steps,
        "Episode Rewards": episode_reward,
        "Average Rewards": episode_reward / num_steps,
        "Epsilon": epsilon,
        "Elapsed": end_time - start_time,
        "Total Elapsed": end_time - total_start_time
    }
    episode_log.update(env_perf)
    print(episode_log)

    training_logs.append(episode_log)
    if episode > 0 and episode % save_interval == 0 or episode == num_episode - 1:        
        save_training_logs(episode==0)        
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_dir + "/" + str(episode))

env.close()