from collections import deque
import random
import torch
from torch.distributions import Normal
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SACAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_size,
            max_action,
            actor,
            critic_1,
            critic_2,
            critic_1_target,
            critic_2_target,
            gamma=.99,
            buffer_size=100000,
            batch_size=1,
            alpha=0.1,
            recurrent=False
        ):
        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.actor = actor
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.0003)

        self.critic1 = critic_1
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=0.0003)
        self.target_critic1 = critic_1_target

        self.critic2 = critic_2
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=0.0003)
        self.target_critic2 = critic_2_target

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003)

        self.max_action = max_action
        self.recurrent = recurrent
        if self.recurrent:
            hx = torch.zeros(self.batch_size, hidden_size)
            cx = torch.zeros(self.batch_size, hidden_size)
            self.hidden_state = (hx, cx)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.detach().numpy()[0]
        return action

    def update(self, batch_size=256, discount=0.99, tau=0.005, alpha_lr=0.0003):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        # Update critic networks
        with torch.no_grad():
            next_action_mean, next_action_std = self.actor(next_state)
            next_action_dist = Normal(next_action_mean, next_action_std)
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.optimizer_critic1.zero_grad()
        critic1_loss.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        critic2_loss.backward()
        self.optimizer_critic2.step()

        # Update actor network
        action_mean, action_std = self.actor(state)
        action_dist = Normal(action_mean, action_std)
        log_prob = action_dist.log_prob(action)
        q_value = torch.min(self.critic1(state, action), self.critic2(state, action))
        actor_loss = (self.alpha * log_prob - q_value).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _sample_next_action_log_prob(self, next_state):
        next_action, next_log_prob = self.actor.sample(next_state)
        return next_action, next_log_prob

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))