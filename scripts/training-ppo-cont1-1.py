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
    "obstacle_preset": 4,
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

from rl_algorithms2.ppo_continuous import PPO
ppo = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    action_range=max_action,
    hidden_dim=hidden_dim,
)

from datetime import datetime
model_type = "ppo_cont1"
train_id = datetime.now().strftime("%y%m%d%H%M%S")
model_dir = "models/" + model_type + "-" + train_id

# %% Training
import csv, os
training_logs = []
def save_training_logs(writeheader=False):
    os.makedirs("logs/train", exist_ok=True)
    train_log_path = "logs/train/" + model_type + "-" + train_id + ".csv"
    exist = os.path.exists(train_log_path)
    with open(train_log_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=episode_log.keys())
        if not exist:
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
AUTO_ENTROPY=True
DETERMINISTIC = False
max_epsilon = 1
min_epsilon = 0.05
decay_rate = 0.0005
num_episode = 100000
save_interval = 100
avg_reward_step_interval = 100
curr_step = 0
batch_size = 1
update_batch = 100
total_start_time = time.time()

epsilon_steps = [1 - (i+1) * .05 for i in range(19)]
print(epsilon_steps)

ppo.init_buffer()
for episode in range(num_episode):
    print('Episode', episode+1)
    start_time = time.time()
    num_steps = 0
    episode_reward = 0
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    #epsilon = epsilon_steps[int(episode / (num_episode / len(epsilon_steps)))]

    while True: # Use config["duration"] to truncate
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = ppo.choose_action(obs, deterministic=False)
        #print(action)

        next_obs, reward, done, truncated, info = env.step(action)

        # Update PPO
        ppo.append_buffer(obs, action, reward, done)
        if (curr_step+1) % update_batch == 0:
            ppo.compute_update(next_obs, done)

        #time_step_rewards.append(reward)
        curr_step += 1
        #if curr_step % avg_reward_step_interval == 0:
        #    save_avg_rewards()
        #    curr_step = 0

        

        obs = next_obs
        num_steps += 1
        episode_reward += reward
        #env.render() # Note: Do not render during training

        if done or truncated:
            obs, info = env.reset()
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
    print(episode_log)

    training_logs.append(episode_log)
    if episode > 0 and episode % save_interval == 0 or episode == num_episode - 1:        
        save_training_logs(episode==0)
        os.makedirs(model_dir, exist_ok=True)
        ppo.save_model(model_dir + "/" + str(episode))

env.close()