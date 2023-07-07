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
    "duration": 500,
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
continue_train = True
if continue_train:
    train_id = "230705191035" # Put the train_id and episode here !!!
    episode = "10000"
    model_dir = "../../data/models/" + model_type + "-" + train_id
    trainer.load_model(model_dir + "/" + episode)
    trainer.policy_net.train()
    trainer.soft_q_net1.train()
    trainer.soft_q_net2.train()
    episode = int(episode)
else:
    train_id = datetime.now().strftime("%y%m%d%H%M%S")
    model_dir = "models/" + model_type + "-" + train_id
    episode = 0

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

obs, info = env.reset()
done = truncated = False

import time
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
AUTO_ENTROPY=True
DETERMINISTIC = False
max_epsilon = 0.5
min_epsilon = 0.05
decay_rate = 0.0005
num_episode = 20000
save_interval = 100
update_interval_episode = 2
non_update_episode_count = -1
batch_size = 2 # Episode
min_batch_size = 4
max_batch_size = 4
sequence_length = 25
min_seq_len = 16
update_itr = 2
total_start_time = time.time()

episode_state = []
episode_action = []
episode_last_action = []
episode_rewards = []
episode_next_state = []
episode_done = []

def get_action(agent, obs, last_action, hidden_in):
    with torch.no_grad():
        action, hidden_out = agent.policy_net.get_action(obs, last_action, hidden_in, deterministic=DETERMINISTIC)
    return action, hidden_out

updated = False
while episode < num_episode:
    print('Episode', episode+1)
    obs, info = env.reset()

    start_time = time.time()
    num_steps = 0
    episode_reward = 0
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # For start of batch
    last_action = np.zeros(env.action_space.shape)
    # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
    #print("Reset hidden state...")
    hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                  torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))

    while True: # Use config["duration"] to truncate
        hidden_in = hidden_out

        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action, hidden_out = get_action(trainer, obs, last_action, hidden_in)
        #print(action)

        next_obs, reward, done, truncated, info = env.step(action)

        if len(episode_state) == 0:
            # Initial hidden state
            ini_hidden_in = hidden_in
            ini_hidden_out = hidden_out

        episode_state.append(obs)
        episode_action.append(action)
        episode_last_action.append(last_action)
        episode_rewards.append(reward)
        episode_next_state.append(next_obs)
        episode_done.append(done)

        # Sequence length is shorter than max steps
        if len(episode_state) == sequence_length:
            #print('Push to replay buffer...')
            replay_buffer.push(
                ini_hidden_in,
                ini_hidden_out,
                episode_state,
                episode_action,
                episode_last_action,
                episode_rewards,
                episode_next_state,
                episode_done)
            # Reset - note: point to new empty list
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_rewards = []
            episode_next_state = []
            episode_done = []

        obs = next_obs
        last_action = action

        num_steps += 1
        episode_reward += reward
        #env.render() # Note: Do not render during training

        if done or truncated:
            env_perf = env.get_performance()
            break

    if len(episode_state) >= min_seq_len:
        # Padding and push
        while len(episode_state) < sequence_length:
            episode_state.append(obs)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_rewards.append(reward)
            episode_next_state.append(next_obs)
            episode_done.append(done)
        replay_buffer.push(
            ini_hidden_in,
            ini_hidden_out,
            episode_state,
            episode_action,
            episode_last_action,
            episode_rewards,
            episode_next_state,
            episode_done)
    # Reset - note: point to new empty list
    episode_state = []
    episode_action = []
    episode_last_action = []
    episode_rewards = []
    episode_next_state = []
    episode_done = []

    # Minimum interval between update
    if len(replay_buffer) >= min_batch_size and non_update_episode_count >= update_interval_episode:
        batch_size = min(int(len(replay_buffer) / 2), max_batch_size)        
        # Update agent
        for i in range(update_itr):
            print("Update SAC-LSTM... Batch size:", batch_size, 'from:', len(replay_buffer))
            updated = trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim,
                                     min_seq_size=None) or updated
            print('Sampled min sequence len:', trainer.min_seq_len)
        if updated:
            non_update_episode_count = 0
            updated = False
        else:
            print('Min sequence length not fulfilled.')
            non_update_episode_count += 1
    else:
        non_update_episode_count += 1

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
    if episode > 0 and (episode+1) % save_interval == 0 or episode == num_episode - 1:
        save_training_logs(episode+1==save_interval)
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_dir + "/" + str(episode+1))
    episode += 1

env.close()