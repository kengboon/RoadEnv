'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf

Use Two branch structure as in paper:
Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
https://arxiv.org/pdf/1710.06537.pdf

RSAC-Share in paper:
Recurrent Off-policy Baselines for Memory-based Continuous Control
https://openreview.net/pdf?id=2IJHEByUwY- 
'''

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from .common.buffers import *
from .common.initialize import linear_weights_init
from .common.value_networks import QNetworkBase
from .common.policy_networks import SAC_PolicyNetworkLSTM

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

class DoubleQNetworkLSTMShare(nn.Module):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu):
        super(DoubleQNetworkLSTMShare, self).__init__()
        self.lstm_branch = LSTMBranch(state_space.shape[0], action_space.shape[0], hidden_dim)
        self.q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim, activation)
        self.q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim, activation)

    def forward(self, state, action, last_action, hidden_in):
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        lstm_out, hidden_out = self.lstm_branch(state, last_action, hidden_in)
        q1 = self.q_net1(state, action, lstm_out)
        q2 = self.q_net2(state, action, lstm_out)
        q_min = torch.min(q1, q2)
        q_min = q_min.permute(1,0,2)  # back to same axes as input
        return q_min, hidden_out

class LSTMBranch(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, activation=F.relu):
        super(LSTMBranch, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._activation = activation
        self.linear1 = nn.Linear(self._state_dim + self._action_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, state, last_action, hidden_in):
        x = torch.cat([state, last_action], -1)
        x = self._activation(self.linear1(x))
        x, h = self.lstm(x, hidden_in)
        return x, h

class QNetworkLSTM(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        #self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        #self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action, lstm_branch):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        # branch 1
        fc_branch = torch.cat([state, action], -1) 
        fc_branch = self.activation(self.linear1(fc_branch))
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        return x

class SAC_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.q_net = DoubleQNetworkLSTMShare(state_space, action_space, hidden_dim).to(device)
        self.target_q_net = DoubleQNetworkLSTMShare(state_space, action_space, hidden_dim).to(device)
        self.policy_net = SAC_PolicyNetworkLSTM(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network-LSTM: ', self.q_net)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer = optim.Adam(self.q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def _trim_arrays(self, arrays, min_len=None):
        if min_len is None:
            min_len = min(len(a) for a in arrays)
        return [a[:min_len] for a in arrays], min_len

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2, min_seq_size=None):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state, min_len = self._trim_arrays(state)
        self.min_seq_len = min_len
        if min_seq_size is not None and min_len < min_seq_size:
            return False

        state      = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(self._trim_arrays(next_state, min_len)[0])).to(device)
        action     = torch.FloatTensor(np.array(self._trim_arrays(action, min_len)[0])).to(device)
        last_action     = torch.FloatTensor(np.array(self._trim_arrays(last_action, min_len)[0])).to(device)
        reward     = torch.FloatTensor(np.array(self._trim_arrays(reward, min_len)[0])).unsqueeze(-1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(self._trim_arrays(done, min_len)[0])).unsqueeze(-1).to(device)

        predicted_q_value, _ = self.q_net(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min, _ = self.target_q_net(next_state, new_next_action, action, hidden_out)
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss = self.soft_q_criterion(predicted_q_value, target_q_value.detach())  # detach: no gradients for the variable

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        # Training Policy Function
        predicted_new_q_value, _ = self.q_net(state, new_action, last_action, hidden_in)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )


        # Soft update the target value net
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        #return predicted_new_q_value.mean()
        return True

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path+'_q')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path+'_q', map_location=torch.device(device)))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.policy_net.load_state_dict(torch.load(path+'_policy', map_location=torch.device(device)))
        self.q_net.eval()
        self.policy_net.eval()

replay_buffer_size = 1e6
replay_buffer = ReplayBufferLSTM2(replay_buffer_size)

# hyper-parameters for RL training
max_episodes  = 1000
max_steps   = 20
frame_idx   = 0
batch_size  = 2
explore_steps = 0  # for action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim =512
rewards     = []
model_path = './model/sac_v2_lstm'