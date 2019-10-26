

import numpy as np
import random
import copy
import torch
from d4pg_model import Actor, Critic


Vmax        = 0.7
Vmin        = -0.7
N_ATOMS     = 51
DELTA_Z     = (Vmax - Vmin) / (N_ATOMS - 1)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=42 , device="cpu",
                 LR_ACTOR=1e-4, LR_CRITIC=1e-4, WEIGHT_DECAY=0.0001):

        self.state_size      = state_size
        self.action_size     = action_size
        # Actor Network (w/ Target Network)
        self.actor_local     = Actor(state_size, action_size, seed).to(device)
        self.actor_target    = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local     = Critic(state_size, action_size, seed,n_atoms=N_ATOMS,
                                       v_min=Vmin, v_max=Vmax).to(device)
        self.critic_target    = Critic(state_size, action_size, seed,n_atoms=N_ATOMS,
                                       v_min=Vmin, v_max=Vmax).to(device)

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.noise = OUNoise(action_size) # add OU noise for exploration
        # initialize targets same as original networks
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        self.device = device

    def hard_update(self,target, source):
        """
        Copy network parameters from source to target
            Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
                    
    def act(self, state ,add_noise=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #state = torch.tensor(np.moveaxis(state,3,1)).float().to(device)
        state = torch.tensor(state).float().to(self.device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            #action = self.actor_local(state).cpu().data

        self.actor_local.train()
        action = torch.from_numpy(action) + add_noise*self.noise.noise()
        return np.clip(action,-1.0,1.0)

    
#####################################################################################################
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


####################################################################################################
class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        if is_weight.max() != 0:
            is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)



################################################################################################
# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


