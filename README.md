# RoadEnv
An environment for urban area *autonomous driving* and decision-making - extended from [highway-env](https://github.com/Farama-Foundation/HighwayEnv) ([Permissive License](https://github.com/Farama-Foundation/HighwayEnv/blob/master/LICENSE)).

## Get Started
1. Install [highway-env](https://github.com/Farama-Foundation/HighwayEnv) via pip for all dependencies.
    ```BAT
    pip install highway-env==1.8.1
    ```
2. Clone the repository.
    ```BAT
    git clone https://github.com/kengboon/RoadEnv.git
    ```
3. Sample code.
    ```Python
    # Register environment
    from road_env import register_road_envs
    register_road_envs()
    
    # Make environment
    import gymnasium as gym
    env = gym.make('urban-road-v0', render_mode='rgb_array')
    env.reset()
    ```

## Decision-making

## Citation

## Buy me a ☕
<a href="https://ko-fi.com/woolf42" target="_blank"><img src="https://user-images.githubusercontent.com/5046671/197377067-ce6016ae-6368-47b6-a4eb-903eb7b0af9c.png" width="200" alt="Support me on Ko-fi"/></a>