import torch
import torch.nn as nn
hidden_dim = 128

class RecurrentActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTMCell(state_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        hx, cx = self.lstm(x, hidden)
        x = self.fc(hx)
        return x, (hx, cx)
    
class RecurrentCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTMCell(state_dim + action_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, hidden):
        x = torch.cat([state, action], dim=1)
        hx, cx = self.lstm(x, hidden)
        x = self.fc(hx)
        return x, (hx, cx)