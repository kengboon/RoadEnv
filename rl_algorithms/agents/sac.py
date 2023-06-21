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
            self.actor = RecurrentActor(state_dim, action_dim, max_action).to(device)
            self.critic = RecurrentCritic(state_dim, action_dim).to(device)
            self.value_net = RecurrentCritic(state_dim, 0).to(device)
            self.target_value_net = RecurrentCritic(state_dim, 0).to(device)
        else:
            self.actor = SoftActor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)
            self.value_net = Critic(state_dim, 0).to(device)
            self.target_value_net = Critic(state_dim, 0).to(device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
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
            expected_q_value, _ = self.critic(state_batch, action_batch, self.hidden_state)
            expected_value = self.value_net(state_batch, None)
            new_action, log_prob, z, mean, log_std = self.actor(state_batch)
            target_value = self.target_value_net(next_state_batch, None)
        else:
            expected_q_value = self.critic(state_batch, action_batch)
            expected_value = self.value_net(state_batch, None)
            new_action, log_prob, z, mean, log_std = self.actor(state_batch)
            target_value = self.target_value_net(next_state_batch, None)
        next_q_value = reward_batch + (1 - done_batch) * self.gamma * target_value
        q_value_loss = F.mse_loss(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.critic(state_batch, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = F.mse_loss(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = 1e-3 * mean.pow(2).mean()
        std_loss = 1e-3 * log_std.pow(2).mean()
        z_loss = 1e-3 * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.critic_optimizer.zero_grad()
        q_value_loss.backward()
        self.critic_optimizer.step()

        self.value_net_optimizer.zero_grad()
        value_loss.backward()
        self.value_net_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def _sample_next_action_log_prob(self, next_state):
        next_action, next_log_prob = self.actor.sample(next_state)
        return next_action, next_log_prob

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
            self.critic.load_state_dict(torch.load(os.path.join(dir, "critic.pth"), map_location=torch.device(device)))
            self.value_net.load_state_dict(torch.load(os.path.join(dir, "value_net.pth"), map_location=torch.device(device)))
            self.target_value_net.load_state_dict(self.value_net.state_dict())

    def load_model(self, dir, device=None, train=False):
        # Only load actor
        self.actor = torch.load(os.path.join(dir, "actor.pth"), map_location=torch.device(device))
        if not train:
            self.eval()
        else:
            self.critic = torch.load(os.path.join(dir, "critic.pth"), map_location=torch.device(device))
            self.value_net = torch.load(os.path.join(dir, "value_net.pth"), map_location=torch.device(device))
            self.target_value_net = torch.load(os.path.join(dir, "value_net.pth"), map_location=torch.device(device))

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(dir, "critic.pth"))
        torch.save(self.value_net.state_dict(), os.path.join(dir, "value_net.pth"))

    def save_model(self, dir):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.actor, os.path.join(dir, "actor.pth"))
        torch.save(self.critic, os.path.join(dir, "critic.pth"))
        torch.save(self.value_net, os.path.join(dir, "value_net.pth"))