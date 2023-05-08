# Hide Pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from gymnasium.envs.registration import register

def register_road_envs():
    register(
        id='urban-road-v0',
        entry_point='road_env.envs:UrbanRoadEnv',
    )