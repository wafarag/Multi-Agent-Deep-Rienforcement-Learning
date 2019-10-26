import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Actor Model Size
Actor_HL1 = 256        # 1st Hidden layer Size
Actor_HL2 = 128        # 2nd Hidden Layer Size

# Critic Model Size
Critic_HL1 = 256       # 1st Hidden layer Size
Critic_HL2 = 128       # 2nd Hidden layer Size

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=Actor_HL1, fc2_units=Actor_HL2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """    
        super(Actor, self).__init__()  
        self.seed  = torch.manual_seed(seed)
        self.fc1   = nn.Linear(state_size, fc1_units)
        self.fc2   = nn.Linear(fc1_units, fc2_units)
        self.fc3   = nn.Linear(fc2_units, action_size)
        self.b2    = nn.BatchNorm1d(action_size)
        self.tanh  = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.b2(x)
        x = self.tanh(x)
        return x        

##########################################################################################
class Critic(nn.Module):
    """Critic (distribution) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=Critic_HL1,
                 fc2_units=Critic_HL2, n_atoms=51, v_min=-1, v_max=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimesion of each action
            seed (int): Random seed
        """    
        super(Critic, self).__init__()
        self.seed   = torch.manual_seed(seed)
        self.fc1    = nn.Linear(state_size, fc1_units)
        self.fc2    = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3    = nn.Linear(fc2_units, n_atoms)
        delta       = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max+delta , delta))
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state,action):
        xs = F.leaky_relu(self.fc1(state))
        x  = torch.cat((xs, action), dim=1)
        x  = F.leaky_relu(self.fc2(x))
        x  = self.fc3(x)
        return x

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res     = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)




