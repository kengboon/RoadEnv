## Disclaimers
The implementations in this directory are from [quantumiracle/Popular-RL-Algorithms](https://github.com/quantumiracle/Popular-RL-Algorithms), **no license** was provided.

The code here only aims to support [demonstrations](https://github.com/kengboon/RoadEnv/tree/main/scripts) on how can an RL agent interacts with the environment.

### General usage
```Python
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
```
