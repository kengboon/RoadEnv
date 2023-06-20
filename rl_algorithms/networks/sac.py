import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, state):
        # Forward pass
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -1, 1)

        # Get action
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2))
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, mean, log_std