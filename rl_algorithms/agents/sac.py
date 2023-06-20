import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from ..networks import *
from ..objects import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SACAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            hidden_size,
            lr_actor=0.001,
            lr_critic=0.001,
            lr_valuenet=0.001,
            gamma=.99,
            tau=0.001,
            buffer_size=100000,
            batch_size=1,
            alpha=0.1,
            recurrent=False
        ):
        if recurrent:
            pass
        else:
            self.actor = SoftActor(state_dim, action_dim, max_action).to(device)
            self.critic1 = Critic(state_dim, action_dim).to(device)
            self.critic2 = Critic(state_dim, action_dim).to(device)
            self.value_net = Critic(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_valuenet)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(maxlen=self.buffer_size)
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.recurrent = recurrent
        if self.recurrent:
            hx = torch.zeros(self.batch_size, hidden_size)
            cx = torch.zeros(self.batch_size, hidden_size)
            self.hidden_state = (hx, cx)

    def flatten_state(self, state):
        return state.flatten()

    def get_action(self, state):
        state = Variable(torch.from_numpy(self.flatten_state(state)).float()).to(device)
        with torch.no_grad():
            if self.recurrent:
                action, self.hidden_state = self.actor(state.unsqueeze(0), self.hidden_state)
                action = action[0]
            else:
                action, _ = self.actor(state)
        action = action.cpu().detach().numpy()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        state_batch = Variable(torch.from_numpy(np.array(state_batch)).float()).to(device)
        action_batch = Variable(torch.from_numpy(np.array(action_batch)).float()).to(device)
        reward_batch = Variable(torch.from_numpy(np.array(reward_batch)).float()).to(device)
        next_state_batch = Variable(torch.from_numpy(np.array(next_state_batch)).float()).to(device)
        done_batch = Variable(torch.from_numpy(np.array(done_batch)).float()).to(device)

        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state_batch)
            q1_next = self.critic1(next_state_batch, next_action)
            q2_next = self.critic2(next_state_batch, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_action_log_prob
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next

        q1 = self.critic1(state_batch, action_batch)
        q2 = self.critic2(state_batch, action_batch)

        critic1_loss = F.mse_loss(q1, next_q_value)
        critic2_loss = F.mse_loss(q2, next_q_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_action, log_prob = self.actor(state_batch)
        q1_new = self.critic1(state_batch, new_action)
        q2_new = self.critic2(state_batch, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        value_net_loss = F.mse_loss(self.value_net(state_batch, action_batch), min_q_new.detach())

        self.value_net_optimizer.zero_grad()
        value_net_loss.backward()
        self.value_net_optimizer.step()

        # Update target value network
        for target_param, param1, param2 in zip(
            self.value_net.parameters(), self.critic1.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(self.tau * param1.data + (1 - self.tau) * param2.data)

    def _sample_next_action_log_prob(self, next_state):
        next_action, next_log_prob = self.actor.sample(next_state)
        return next_action, next_log_prob

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((self.flatten_state(state), action, reward, self.flatten_state(next_state), done))

    def get_log(self):
        return {}

    def load(self, dir, device=None):
        self.actor.load_state_dict(torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device)))
        self.critic1.load_state_dict(torch.load(os.path.join(dir, "critic1.pth"), map_location=torch.device(device)))
        self.critic2.load_state_dict(torch.load(os.path.join(dir, "critic2.pth"), map_location=torch.device(device)))
        self.value_net.load_state_dict(torch.load(os.path.join(dir, "value_net.pth"), map_location=torch.device(device)))

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(dir, "actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(dir, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(dir, "critic2.pth"))
        torch.save(self.value_net.state_dict(), os.path.join(dir, "value_net.pth"))