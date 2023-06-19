from collections import deque
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from ..networks import *

class DDPGAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            hidden_size,
            lr_actor=0.001,
            lr_critic=0.001,
            gamma=.99,
            tau=0.001,
            buffer_size=10000,
            batch_size=1,
            recurrent=False
        ):
        if recurrent:
            self.actor = RecurrentActor(state_dim, action_dim, max_action)
            self.actor_target = RecurrentActor(state_dim, action_dim, max_action)
            self.critic = RecurrentCritic(state_dim, action_dim)
            self.critic_target = RecurrentCritic(state_dim, action_dim)
        else:
            self.actor = Actor(state_dim, action_dim, max_action)
            self.actor_target = Actor(state_dim, action_dim, max_action)
            self.critic = Critic(state_dim, action_dim)
            self.critic_target = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.recurrent = recurrent
        if self.recurrent:
            hx = torch.zeros(self.batch_size, hidden_size)
            cx = torch.zeros(self.batch_size, hidden_size)
            self.hidden_state = (hx, cx)

    def flatten_state(self, state):
        return state.flatten()

    def get_action(self, state):
        state = Variable(torch.from_numpy(self.flatten_state(state)).float())
        with torch.no_grad():
            if self.recurrent:
                action, self.hidden_state = self.actor(state, self.hidden_state)
            else:
                action = self.actor(state)
        action = action.detach().numpy()
        return action
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = Variable(torch.from_numpy(np.array(state_batch)).float())
        action_batch = Variable(torch.from_numpy(np.array(action_batch)).float())
        reward_batch = Variable(torch.from_numpy(np.array(reward_batch)).float())
        next_state_batch = Variable(torch.from_numpy(np.array(next_state_batch)).float())
        done_batch = Variable(torch.from_numpy(np.array(done_batch)).float())

        # Update critic
        if self.recurrent:
            Q_values, _ = self.critic(state_batch, action_batch, self.hidden_state)
            with torch.no_grad():
                next_actions, _ = self.actor_target(next_state_batch, self.hidden_state)
                next_Q_values, _ = self.critic_target(next_state_batch, next_actions, self.hidden_state)
        else:
            Q_values = self.critic(state_batch, action_batch)
            with torch.no_grad():
                next_actions = self.actor_target(next_state_batch)
                next_Q_values = self.critic_target(next_state_batch, next_actions)
        target_Q_values = reward_batch + (1 - done_batch) * self.gamma * next_Q_values
        critic_loss = F.mse_loss(Q_values, target_Q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        if self.recurrent:
            actions, _ = self.actor(state_batch, self.hidden_state)
            actor_loss = -self.critic(state_batch, actions, self.hidden_state)[0].mean()
        else:
            actions = self.actor(state_batch)
            actor_loss = -self.critic(state_batch, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((self.flatten_state(state), action, reward, self.flatten_state(next_state), done))