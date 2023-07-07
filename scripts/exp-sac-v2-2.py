# %% Working directory
import sys, os
if os.path.exists('road_env'):
    sys.path.append('.')
else:
    sys.path.append('..')

# %% Register environment
from road_env import register_road_envs
register_road_envs()

# %% Make environment
import gymnasium as gym
env = gym.make('urban-road-v0', render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
print(f'State dim: {state_dim}, Action dim: {action_dim}, Action range: ({-max_action}, {max_action})')

# %% Make DRL Agent
hidden_dim = 512
from rl_algorithms2.sac_v2 import SAC_Trainer, replay_buffer
agent = SAC_Trainer(
    state_dim=state_dim,
    action_dim=action_dim,
    action_range=max_action,
    hidden_dim=hidden_dim,
    replay_buffer=replay_buffer
)

# %% Load trained model
model_type = 'sac_v2'
train_id = '230704203226'
episode = '10000'
model_dir = f'../../data/models/{model_type}-{train_id}/{episode}'
agent.load_model(model_dir)
print(f'Model loaded from {model_dir}')

# %% Define evaluation function
from enum import Enum
import time

class OCCLUSION(Enum):
    Low = 1
    Medium = 2
    High = 3

def run_eval(env,
             occlusion_level,
             agent,
             num_episode=20,
             max_step=999,
             auto_randseed=True,
             render=False):
    if isinstance(occlusion_level, OCCLUSION):
        occlusion_level = occlusion_level._value_
    env.configure({
        'duration': max_step,
        'obstacle_preset': occlusion_level
    })
    infos = []
    start_time = time.time()
    for episode in range(num_episode):
        step = 0
        episode_reward = 0

        if auto_randseed:
            env.configure({
                'random_seed': int(episode * occlusion_level)
            })
        obs, info = env.reset()
        while True: # Number of steps controlled by env.config['duration]
            action = agent.policy_net.get_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if render:
                env.render()

            d = {
                "episode": episode,
                "step": step,
                "reward": reward,
            }
            d.update(info)
            infos.append(d)

            step += 1
            episode_reward += reward
            if done or truncated:
                break
        print(f'Episode: {episode+1}, Steps: {step}')
    end_time = time.time()
    print(f'Total elapsed: {end_time-start_time: .3f}')
    return infos

# %% Run evaluation
from datetime import datetime
import os
import pandas as pd

test_id = datetime.now().strftime('%y%m%d%H%M%S')
num_episode = 50
max_step = 999
road_length = 1000
render=True
print(f'Test ID: {test_id}')

for occ in OCCLUSION:
    print(occ)
    info = run_eval(env, occ, agent,
                    num_episode=num_episode,
                    max_step=max_step,
                    render=render)
    info = pd.DataFrame(info) # Convert into dataframe

    log_path = f"../../data/logs/test/{model_type}-{train_id}/{test_id}-{occ._value_}.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    info.to_csv(log_path, index=False)
    print(f'Saved to {log_path}')