import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from ..networks import *
from ..objects import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TD3Agent:
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
            policy_noise=.2,
            policy_freq=2,
            recurrent=False,
        ):
        if recurrent:
            self.actor = RecurrentActor(
                state_dim, action_dim, max_action,
                batch_size=batch_size, hidden_dim=hidden_size).to(device)
            self.actor_target = RecurrentActor(
                state_dim, action_dim, max_action,
                batch_size=batch_size, hidden_dim=hidden_size).to(device)
            self.critic1 = RecurrentCritic(
                state_dim, action_dim,
                batch_size=batch_size, hidden_dim=hidden_size).to(device)
            self.critic1_target = RecurrentCritic(
                state_dim, action_dim,
                batch_size=batch_size, hidden_dim=hidden_size).to(device)
            self.critic2 = RecurrentCritic(
                state_dim, action_dim,
                batch_size=batch_size, hidden_dim=hidden_size).to(device)
            self.critic2_target = RecurrentCritic(
                state_dim, action_dim,
                batch_size=batch_size, hidden_dim=hidden_size).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
            self.critic1 = Critic(state_dim, action_dim).to(device)
            self.critic1_target = Critic(state_dim, action_dim).to(device)
            self.critic2 = Critic(state_dim, action_dim).to(device)
            self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(maxlen=self.buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.step_count = 0
        self.recurrent = recurrent
        self.actor_losses = []
        self.critic_losses = []

    def flatten_state(self, state):
        return state.flatten()

    def get_action(self, state):
        state = Variable(torch.from_numpy(self.flatten_state(state)).float()).to(device)
        with torch.no_grad():
            if self.recurrent:
                action, self.hidden_state = self.actor(state.unsqueeze(0), None)
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
            with torch.no_grad():
                next_actions, _ = self.actor_target(next_state_batch, self.hidden_state)
        else:
            with torch.no_grad():
                next_actions = self.actor_target(next_state_batch)
        noise = action_batch.data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-0.5, 0.5)
        next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action).to(device)

        if self.recurrent:
            with torch.no_grad():
                target_Q1, _ = self.critic1_target(next_state_batch, next_actions, self.hidden_state)
                target_Q2, _ = self.critic2_target(next_state_batch, next_actions, self.hidden_state)
        else:
            with torch.no_grad():
                target_Q1 = self.critic1_target(next_state_batch, next_actions)
                target_Q2 = self.critic2_target(next_state_batch, next_actions)
        target_Q_values = torch.min(target_Q1, target_Q2)
        target_Q_values = reward_batch + (1 - done_batch) * self.gamma * target_Q_values

        if self.recurrent:
            current_Q1, _ = self.critic1(state_batch, action_batch, self.hidden_state)
            current_Q2, _ = self.critic2(state_batch, action_batch, self.hidden_state)
        else:
            current_Q1 = self.critic1(state_batch, action_batch)
            current_Q2 = self.critic2(state_batch, action_batch)
        critic_loss = F.mse_loss(current_Q1, target_Q_values) + F.mse_loss(current_Q2, target_Q_values)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Delay policy update
        if self.step_count % self.policy_freq == 0:
            self.step_count = 0

            # Update actor
            if self.recurrent:
                actions, self.hidden_state = self.actor(state_batch, None)
                actor_loss = -self.critic1(state_batch, actions, self.hidden_state)[0].mean()
            else:
                actions = self.actor(state_batch)
                actor_loss = -self.critic1(state_batch, actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.step_count += 1

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((self.flatten_state(state), action, reward, self.flatten_state(next_state), done))

    def get_log(self):
        return {}

    def eval(self):
        self.train(False)

    def train(self, mode: bool=True):
        self.actor.train(mode=mode)

    def load(self, dir, device=None, train=False):
        self.actor.load_state_dict(torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device)))
        if not train:
            self.eval()
        else:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic1.load_state_dict(torch.load(os.path.join(dir, "critic1.pth"), map_location=torch.device(device)))
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2.load_state_dict(torch.load(os.path.join(dir, "critic2.pth"), map_location=torch.device(device)))
            self.critic2_target.load_state_dict(self.critic1.state_dict())

    def load_model(self, dir, device=None, train=False):
        self.actor = torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device))
        if not train:
            if self.recurrent:
                self.actor.reset_hidden()
            self.eval()
        else:
            self.actor_target = torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device))
            self.critic1 = torch.load(os.path.join(dir, "critic1.pth"), map_location=torch.device(device))
            self.critic1_target = torch.load(os.path.join(dir, "critic1.pth"), map_location=torch.device(device))
            self.critic2 = torch.load(os.path.join(dir, "critic2.pth"), map_location=torch.device(device))
            self.critic2_target = torch.load(os.path.join(dir, "critic2.pth"), map_location=torch.device(device))

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(dir, "actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(dir, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(dir, "critic2.pth"))

    def save_model(self, dir):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.actor, os.path.join(dir, "actor.pth"))
        torch.save(self.critic1, os.path.join(dir, "critic1.pth"))
        torch.save(self.critic2, os.path.join(dir, "critic2.pth"))