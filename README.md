# RoadEnv
An environment for urban area *autonomous driving* and decision-making - extended from [highway-env](https://github.com/Farama-Foundation/HighwayEnv).

<p align="center"><img src="https://github.com/kengboon/RoadEnv/assets/5046671/8a9c1c2b-3f62-4266-ab9c-cd0054267f45"/></p>

### Scenarios
- Roadside obstacles
- Pedestrians cross the road

### Objective
- Pedestrian collision avoidance

### Observation
- [Line of sight + kinematics](https://github.com/kengboon/RoadEnv/blob/main/road_env/envs/common/observation.py#L631)

## Get Started
1. Install [highway-env](https://github.com/Farama-Foundation/HighwayEnv) via pip for all dependencies.
    ```BAT
    pip install highway-env==1.8.1
    ```
2. Clone the repository.
    ```BAT
    git clone https://github.com/kengboon/RoadEnv.git
    ```
3. Sample code (see [scripts](https://github.com/kengboon/RoadEnv/tree/main/scripts)).
    ```Python
    # Register environment
    from road_env import register_road_envs
    register_road_envs()
    
    # Make environment
    import gymnasium as gym
    env = gym.make('urban-road-v0', render_mode='rgb_array')
    env.reset()
    ```

## Usage
```Python
# Register environment
from road_env import register_road_envs
register_road_envs()

# Make environment
import gymnasium as gym
env = gym.make('urban-road-v0', render_mode='rgb_array')

# Configure parameters (example)
env.configure({
    "random_seed": None,
    "duration": 60,
})

obs, info = env.reset()

# Graphic display
import matplotlib.pyplot as plt
plt.imshow(env.render())

# Execution
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
    env.render() # Update graphic
```

See [more examples](https://github.com/kengboon/RoadEnv/tree/main/scripts).

## Citation

## Buy me a ☕
<a href="https://ko-fi.com/woolf42" target="_blank"><img src="https://user-images.githubusercontent.com/5046671/197377067-ce6016ae-6368-47b6-a4eb-903eb7b0af9c.png" width="200" alt="Support me on Ko-fi"/></a>
