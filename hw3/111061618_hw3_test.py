import gym
import gym_multi_car_racing
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from torchvision import transforms as T
import cv2
import numpy as np
import warnings
import math
import random
import pickle
from pickle import UnpicklingError
from gym.envs.box2d.car_dynamics import Car
from pathlib import Path
import itertools

import matplotlib.pyplot as plt


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6



### memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, action, reward, n_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, action, reward, n_obs, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, n_obs, done = map(np.stack, zip(*batch))
        return obs, action, reward, n_obs, done
    def save(self):
        print(f"Saving replay buffer to {path}.")
        save_dir = Path("memory/")
        if not save_dir.exists():
            save_dir.mkdir()
        with open((save_dir/path).with_suffix(".pkl"), "wb") as fp:
            pickle.dump(self.buffer, fp)
    def load(self):
        try:
            with open(path, "rb") as fp:
                mem = pickle.load(fp)
                assert len(mem) == self.capacity, "Capacity of the replay buffer and length of the loaded memory don't match!"
                self.buffer = mem
            print(f"Loaded saved replay buffer from {path}.")
        except UnpicklingError:
            raise TypeError("This file doesn't contain a pickled list!")

    def __len__(self):
        return len(self.buffer)

### updates
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


### networks
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        ## new
        #self.cv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2)
        #self.cv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)    


        # Q1 architecture
        self.linear1 = nn.Linear(4624 + num_actions, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(4624 + num_actions, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        state = self.net(state)
        xu = torch.cat([state, action], 1)

        x1 = F.relu(torch.nn.functional.linear(xu, self.linear1.weight.clone(), self.linear1.bias))
        x1 = F.relu(torch.nn.functional.linear(x1, self.linear2.weight.clone(), self.linear2.bias))
        #x1 = self.linear3(x1)
        x1 = torch.nn.functional.linear(x1, self.linear3.weight.clone(), self.linear3.bias)

        #x2 = F.relu(self.linear4(xu))
        #x2 = F.relu(self.linear5(x2))
        #x2 = self.linear6(x2)

        x2 = F.relu(torch.nn.functional.linear(xu, self.linear4.weight.clone(), self.linear4.bias))
        x2 = F.relu(torch.nn.functional.linear(x2, self.linear5.weight.clone(), self.linear5.bias))
        x2 = torch.nn.functional.linear(x2, self.linear6.weight.clone(), self.linear6.bias)

        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4624,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
        )
        #self.linear1 = nn.Linear(num_inputs, 256)
        #self.linear2 = nn.Linear(256, 256)


        #check where's wrong
        self.cv1 =  nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(4624,256)
        self.linear2 = nn.Linear(256,256)


        self.mean_linear = nn.Linear(256, num_actions)
        self.log_std_linear = nn.Linear(256, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        #print(f"state in gaussian:{state.shape}")
        #x = self.net(state)
        x = F.relu(self.cv1(state))
        #print(f"step1:{x.shape}")
        x = F.relu(self.cv2(x))
        #print(f"step2:{x.shape}")
        if x.dim() == 3:
            x = torch.unsqueeze(x,0)
        x = torch.flatten(x,1)
        #print(f"flatten:{x.shape}")
        x = F.relu(self.linear1(x))
        #print(f"step3:{x.shape}")
        x = F.relu(self.linear2(x))
        #print(f"step4:{x.shape}")


        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True) #this one if sum(0) train wrong else test wrong
        return action, log_prob, torch.tanh(mean)


### agent
class Agent(object):
    def __init__(self, action_space):
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 0.0003

        self.target_update_interval = 1

        self.input_dim = 32
        self.hidden_size = 256
        self.bs = 256

        self.critic = QNetwork(self.input_dim, action_space.shape[0])
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(self.input_dim, action_space.shape[0])
        hard_update(self.critic_target, self.critic)


        self.policy = GaussianPolicy(self.input_dim, action_space.shape[0])
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def choose_action(self, obs, eval=False):
        #print(f"obs in choose before changing:{obs.shape}")
        obs = torch.FloatTensor(obs).unsqueeze(0)
        #print(f"obs in choose after changing:{obs.shape}")
        if eval == False:
            action,_,_ = self.policy.sample(obs)
            action = action.detach().numpy()
            return action[0]
        else:
            obs = torch.FloatTensor(obs).unsqueeze(0)
            #print(f"obs in choose after another changing:{obs.shape}")
            _,_,action = self.policy.sample(obs)
            action = action.detach().numpy()
            #print(f"whole action:{action}")
            return action

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1)
        #check size
        state_batch = torch.unsqueeze(state_batch, 1)
        next_state_batch = torch.unsqueeze(next_state_batch, 1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)

            #print(qf2_next_target.shape)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            #print(f"min_qf_next_target:{min_qf_next_target.shape}")
            #print("reward_batch:",reward_batch.shape)
            #print(f"mask_batch:{mask_batch.shape}")
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        #print(next_q_value.shape)
        ### ori
        """
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2] 這行更新好像有問題
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2] 這行更新好像有問題

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward() #這行會報錯
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        """
        ### try
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_optim.step()

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        self.critic_optim.zero_grad()
        qf2_loss.backward(retain_graph=True) #這行會報錯
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()



        alpha_loss = torch.tensor(0.)


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    def save_model(self):
        ## save model
        state_dict1 = self.policy.state_dict()
        torch.save(state_dict1, "policy_model.pth")

        state_dict2 = self.critic.state_dict()
        torch.save(state_dict2, "critic_model.pth")


    def load_model(self, actor_path = "policy_model.pth", critic_path = "critic_model.pth"):
        #load model
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

def apply(func, M, d=0):
    tList = [func(m) for m in torch.unbind(M, dim=d) ]
    res = torch.stack(tList, dim=d)

    return res
def process_observation(obs):
    cropper = T.Compose([T.ToPILImage(),
                            T.CenterCrop((64,64)),
                            T.ToTensor()])
    converted = torch.from_numpy(obs.copy())
    converted.unsqueeze_(0)
    converted = torch.einsum("nhwc -> nchw", converted)
    return apply(cropper, converted)


env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

###start test
agent = Agent(env.action_space)  
agent.load_model()

avg_r = 0.
episodes = 5
for i in range(episodes):
    print(f"test, {i}/{episodes}")
    obs = env.reset()
    ### do something to obs
    obs = obs[0]
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs[40:80, 30:70]


    ep_r = 0
    done = False
    while not done:
        action = agent.choose_action(obs, True)
        n_obs, reward, done, _ = env.step(action)
        ### do something to n_obs
        n_obs = n_obs[0]
        n_obs = cv2.cvtColor(n_obs, cv2.COLOR_BGR2GRAY)
        n_obs = n_obs[40:80, 30:70]

        ep_r += reward
        obs = n_obs
    avg_r += ep_r[0]
avg_r /= episodes
print("-"*40)
print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_r, 2)}")
print("-"*40)
        
