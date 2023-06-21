import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, batch_size=1, hidden_dim=128):
        super().__init__()
        self.max_action = max_action
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc2 = nn.Linear(int(hidden_dim / 2), action_dim)
        self.reset_hidden()

    def forward(self, state, hidden):
        if hidden is not None:
            self.hidden_state = hidden
        hx, cx = self.hidden_state = self.lstm(state, self.hidden_state)
        x = F.relu(hx)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x) * self.max_action
        return x, self.hidden_state
    
    def reset_hidden(self):
        hx = torch.zeros(self.batch_size, self.hidden_dim)
        cx = torch.zeros(self.batch_size, self.hidden_dim)
        self.hidden_state = (hx, cx)
    
class RecurrentCritic(nn.Module):
    def __init__(self, state_dim, action_dim, batch_size=1, hidden_dim=128):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(state_dim + action_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, int(hidden_dim / 2))
        self.fc2 = nn.Linear(int(hidden_dim / 2), 1)

    def forward(self, state, action, hidden):
        if action is not None:
            x = torch.cat([state, action], dim=1)
        else:
            x = state
        hx, cx = self.lstm(x, hidden)
        x = F.relu(hx)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, (hx, cx)