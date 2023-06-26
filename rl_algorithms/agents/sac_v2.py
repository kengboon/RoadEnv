import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from ..networks import *
from ..objects import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SACv2Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            hidden_size,
            lr_actor=0.001,
            lr_critic=0.001,
            lr_alpha=0.001,
            gamma=.99,
            tau=0.001,
            buffer_size=100000,
            batch_size=1,
            alpha=1,
            target_entropy=-2,
            auto_entropy=True,
            reward_scale=10,
            recurrent=False
        ):
        if recurrent:
            self.actor = RecurrentActor(state_dim, action_dim, max_action).to(device)
            self.critic1 = RecurrentCritic(state_dim, action_dim).to(device)
            self.critic2 = RecurrentCritic(state_dim, action_dim).to(device)
            self.target_critic1 = RecurrentCritic(state_dim, action_dim).to(device)
            self.target_critic2 = RecurrentCritic(state_dim, action_dim).to(device)
        else:
            self.actor = SoftActor(state_dim, action_dim, max_action).to(device)
            self.critic1 = Critic(state_dim, action_dim).to(device)
            self.critic2 = Critic(state_dim, action_dim).to(device)
            self.target_critic1 = Critic(state_dim, action_dim).to(device)
            self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.alpha_optimzer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(maxlen=self.buffer_size)
        self.gamma = gamma
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.auto_entropy = auto_entropy
        self.reward_scale = reward_scale
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
                action = self.actor(state)[0]
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

        if self.recurrent:
            predicted_Q1, _ = self.critic1(state_batch, action_batch, self.hidden_state)
            predicted_Q2, _ = self.critic2(state_batch, action_batch, self.hidden_state)
            new_action, log_prob, z, mean, log_std = self.actor(state_batch)
            new_next_action, next_log_prob, _, _, _ = self.actor(next_state_batch)
        else:
            predicted_Q1 = self.critic1(state_batch, action_batch)
            predicted_Q2 = self.critic2(state_batch, action_batch)
            new_action, log_prob, z, mean, log_std = self.actor(state_batch)
            new_next_action, next_log_prob, _, _, _ = self.actor(next_state_batch)
        reward_batch = self.reward_scale * (reward_batch - reward_batch.mean(dim=0)) / (reward_batch.std(dim=0) + 1e-6)

        # Update alpha wrt entropy
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimzer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimzer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Update critic
        with torch.no_grad():
            if self.recurrent:
                target_Q1, _ = self.target_critic1(next_state_batch, new_next_action)
                target_Q2, _ = self.target_critic2(next_state_batch, new_next_action)
            else:
                target_Q1 = self.target_critic1(next_state_batch, new_next_action)
                target_Q2 = self.target_critic2(next_state_batch, new_next_action)
        target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
        loss_Q1 = F.mse_loss(predicted_Q1, target_Q.detach())
        loss_Q2 = F.mse_loss(predicted_Q2, target_Q.detach())

        self.critic1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic2_optimizer.step()

        # Update actor
        if self.recurrent:
            predicted_Q1, _ = self.critic1(state_batch, new_action, self.hidden_state)
            predicted_Q2, _ = self.critic2(state_batch, new_action, self.hidden_state)
        else:
            predicted_Q1 = self.critic1(state_batch, new_action)
            predicted_Q2 = self.critic2(state_batch, new_action)
        predicted_new_Q = torch.min(predicted_Q1, predicted_Q2)
        actor_loss = (self.alpha * log_prob - predicted_new_Q).mean()

        #mean_loss = 1e-3 * mean.pow(2).mean()
        #std_loss = 1e-3 * log_std.pow(2).mean()
        #z_loss = 1e-3 * z.pow(2).sum(1).mean()

        #actor_loss += mean_loss + std_loss + z_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((self.flatten_state(state), action, reward, self.flatten_state(next_state), done))

    def get_log(self):
        return {}

    def eval(self):
        self.train(False)

    def train(self, mode: bool = True):
        self.actor.train(mode=mode)

    def load(self, dir, device=None, train=False):
        self.actor.load_state_dict(torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device)))
        if not train:
            self.eval()
        else:
            self.critic1.load_state_dict(torch.load(os.path.join(dir, "critic1.pth"), map_location=torch.device(device)))
            self.critic2.load_state_dict(torch.load(os.path.join(dir, "critic2.pth"), map_location=torch.device(device)))
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())

    def load_model(self, dir, device=None, train=False):
        # Only load actor
        self.actor = torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device))
        if not train:
            self.eval()
        else:
            self.critic1 = torch.load(os.path.join(dir, "critic1.pth"), map_location=torch.device(device))
            self.critic2 = torch.load(os.path.join(dir, "critic2.pth"), map_location=torch.device(device))
            self.target_critic1 = torch.load(os.path.join(dir, "critic1.pth"), map_location=torch.device(device))
            self.target_critic2 = torch.load(os.path.join(dir, "critic2.pth"), map_location=torch.device(device))

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