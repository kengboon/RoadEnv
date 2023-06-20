import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from ..networks import *
from ..objects import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            buffer_size=100000,
            batch_size=1,
            recurrent=False
        ):
        if recurrent:
            self.actor = RecurrentActor(state_dim, action_dim, max_action).to(device)
            self.actor_target = RecurrentActor(state_dim, action_dim, max_action).to(device)
            self.critic = RecurrentCritic(state_dim, action_dim).to(device)
            self.critic_target = RecurrentCritic(state_dim, action_dim).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)
            self.critic_target = Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(maxlen=self.buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.recurrent = recurrent
        if self.recurrent:
            hx = torch.zeros(self.batch_size, hidden_size)
            cx = torch.zeros(self.batch_size, hidden_size)
            self.hidden_state = (hx, cx)
        self.actor_losses = []
        self.critic_losses = []

    def flatten_state(self, state):
        return state.flatten()

    def get_action(self, state):
        state = Variable(torch.from_numpy(self.flatten_state(state)).float()).to(device)
        with torch.no_grad():
            if self.recurrent:
                action, self.hidden_state = self.actor(state.unsqueeze(0), self.hidden_state)
                action = action[0]
            else:
                action = self.actor(state)
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
        self.critic_losses.append(critic_loss)

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
        self.actor_losses.append(actor_loss)

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

    def get_log(self):
        avg_actor_loss = sum(self.actor_losses) / len(self.actor_losses)
        avg_critic_loss = sum(self.critic_losses) / len(self.critic_losses)
        self.actor_losses.clear()
        self.critic_losses.clear()
        return {
            "Average Actor Loss": avg_actor_loss.item(),
            "Average Critic Loss": avg_critic_loss.item()
        }

    def load(self, dir, device=None):
        self.actor.load_state_dict(torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device)))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(os.path.join(dir, "critic.pth"), map_location=torch.device(device)))
        self.critic_target.load_state_dict(self.critic.state_dict())

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(dir, "critic.pth"))