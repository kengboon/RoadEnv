"""
Proximal Policy Optimization (PPO) version 1
----------------------------
2 actors and 1 critic
old policy given by old actor, which is delayed copy of actor

To run
------
python tutorial_PPO.py --train/test
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

#####################  hyper parameters  ####################

ENV_NAME = 'Pendulum-v0'  # environment name HalfCheetah-v2 Pendulum-v0
RANDOMSEED = 2  # random seed

EP_MAX = 10000  # total number of episodes for training
EP_LEN = 1000  # total number of steps for each episode
GAMMA = 0.99  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 1024  # update batchsize
A_UPDATE_STEPS = 50  # actor update steps
C_UPDATE_STEPS = 50  # critic update steps
EPS = 1e-8  # numerical residual
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization

###############################  PPO  ####################################

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        # self.linear4.weight.data.uniform_(-init_w, init_w)
        # self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # implementation 1
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        # implementation 2: not dependent on latent features, reference:https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py
        self.log_std = AddBias(torch.zeros(num_actions))  

        self.num_actions = num_actions
        self.action_range = action_range

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))

        mean    = self.action_range * torch.tanh(self.mean_linear(x))

        # implementation 1
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    
        # implementation 2
        zeros = torch.zeros(mean.size())
        if state.is_cuda:
            zeros = zeros.cuda()
        log_std = self.log_std(zeros)

        return mean, log_std
        
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)

        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = normal.sample() 
        action = torch.clamp(action, -self.action_range, self.action_range)
        return action.squeeze(0)

    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return a.numpy()

class PPO(object):
    '''
    PPO class
    '''
    def __init__(self, state_dim, action_dim, action_range=1., hidden_dim=512, a_lr=3e-4, c_lr=3e-4):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.actor_old = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c_lr)
        print(self.actor, self.critic)

    def a_train(self, s, a, adv):
        '''
        Update policy network
        :param s: state
        :param a: action
        :param adv: advantage
        :return:
        '''  
        mu, log_std = self.actor(s)
        pi = Normal(mu, torch.exp(log_std))

        mu_old, log_std_old = self.actor_old(s)
        oldpi = Normal(mu_old, torch.exp(log_std_old))

        # ratio = torch.exp(pi.log_prob(a) - oldpi.log_prob(a))
        ratio = torch.exp(pi.log_prob(a)) / (torch.exp(oldpi.log_prob(a)) + EPS)

        surr = ratio * adv
        if METHOD['name'] == 'kl_pen':
            lam = METHOD['lam']
            kl = torch.distributions.kl.kl_divergence(oldpi, pi)
            kl_mean = kl.mean()
            aloss = -((surr - lam * kl).mean())
        else:  # clipping method, find this is better
            aloss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv))
        self.actor_optimizer.zero_grad()
        aloss.backward()
        self.actor_optimizer.step()

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        Update old policy parameter
        :return: None
        '''
        for p, oldp in zip(self.actor.parameters(), self.actor_old.parameters()):
            oldp.data.copy_(p)


    def c_train(self, cumulative_r, s):
        '''
        Update actor network
        :param cumulative_r: cumulative reward
        :param s: state
        :return: None
        '''
        v = self.critic(s)
        advantage = cumulative_r - v
        closs = (advantage**2).mean()
        self.critic_optimizer.zero_grad()
        closs.backward()
        self.critic_optimizer.step()

    def cal_adv(self, s, cumulative_r):
        '''
        Calculate advantage
        :param s: state
        :param cumulative_r: cumulative reward
        :return: advantage
        '''
        advantage = cumulative_r - self.critic(s)
        return advantage.detach()

    def init_buffer(self):
        self.buffer = {
            "state": [],
            "action": [],
            "reward": [],
            "done": [],
        }

    def append_buffer(self, s, a, r, done):
        self.buffer["state"].append(s)
        self.buffer["action"].append(a)
        self.buffer["reward"].append(r)
        self.buffer["done"].append(done)

    def compute_update(self, s_, done):
        if done:
            v_s_ = 0
        else:
            v_s_ = self.get_v(s_)[0]
        discounted_r = []
        for r, d in zip(self.buffer['reward'][::-1], self.buffer['done'][::-1]):
            v_s_ = r + GAMMA * v_s_ * (1-d)
            discounted_r.append(v_s_)
        discounted_r.reverse
        bs, ba, br = np.vstack(self.buffer['state']), np.vstack(self.buffer['action']), np.array(discounted_r)[:, np.newaxis]
        self.init_buffer()
        self.update(bs, ba, br)

    def update(self, s, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        '''
        s = torch.FloatTensor(s).to(device)     
        a = torch.FloatTensor(a).to(device) 
        r = torch.FloatTensor(r).to(device)   

        adv = self.cal_adv(s, r)
        adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful, not always, minus mean is dangerous

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)    
            
        self.update_old_pi()
 

    def choose_action(self, s, deterministic=False):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        a = self.actor.get_action(s, deterministic)
        return a.detach().cpu().numpy()
    
    def get_v(self, s):
        '''
        Compute value
        :param s: state
        :return: value
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.FloatTensor(s).to(device)  
        # return self.critic(s).detach().cpu().numpy()[0, 0]
        return self.critic(s).squeeze(0).detach().cpu().numpy()


    def save_model(self, path):
        torch.save(self.actor.state_dict(), path+'_actor')
        torch.save(self.critic.state_dict(), path+'_critic')
        torch.save(self.actor_old.state_dict(), path+'_actor_old')

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path+'_actor'))
        self.critic.load_state_dict(torch.load(path+'_critic'))
        self.actor_old.load_state_dict(torch.load(path+'_actor_old'))

        self.actor.eval()
        self.critic.eval()
        self.actor_old.eval()