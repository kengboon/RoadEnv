import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden1 = 256
hidden2 = 128

class SoftActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.mean = nn.Linear(hidden2, action_dim)
        self.log_std = nn.Linear(hidden2, action_dim)

    def forward(self, state, deterministic=True):
        # Forward pass
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -1, 1)

        # Get action
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample().to(device)
        prob = mean if deterministic else mean + std * z
        if deterministic:
            action = torch.tanh(prob) * self.max_action
        else:
            action = torch.tanh(prob) * self.max_action

        normal = Normal(mean, std)
        log_prob = normal.log_prob(prob)\
            - torch.log(1 - action.pow(2))\
            - np.log(self.max_action)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, mean, log_std