from collections import deque
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

class DDPGAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            hidden_size,
            actor,
            actor_target,
            critic,
            critic_target,
            lr_actor=0.001,
            lr_critic=0.001,
            gamma=.99,
            tau=0.001,
            buffer_size=100000,
            batch_size=64
        ):
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.hx = torch.zeros(1, hidden_size)
        self.cx = torch.zeros(1, hidden_size)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state.unsqueeze(0)).squeeze(0)
        return action.detach().numpy()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Update critic
        Q_values = self.critic(state_batch, action_batch)
        next_actions = self.actor_target(next_state_batch)
        next_Q_values = self.critic_target(next_state_batch, next_actions.detach())
        target_Q_values = reward_batch + (1 - done_batch) * self.gamma * next_Q_values
        critic_loss = F.mse_loss(Q_values, target_Q_values.unsqueeze(1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
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
        self.replay_buffer.append((state, action, reward, next_state, done))