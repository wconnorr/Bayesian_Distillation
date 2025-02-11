import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

# Distiller w/ 3D input (images)
class Distiller3D(nn.Module):
  def __init__(self, c, h, w, n_classes, batch_size, inner_lr=.02, inner_momentum=None, conditional_generation=False):
    super(Distiller3D, self).__init__()
    self.conditional_generation = conditional_generation

    self.x = nn.Parameter(torch.randn((batch_size, c, h, w)), True)
    if not conditional_generation:
      self.y = nn.Parameter(torch.randn((batch_size, n_classes)), True)

    # Inner optimizer parameters
    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr), True)
    if inner_momentum is not None:
      self.inner_momentum = nn.Parameter(torch.tensor(inner_momentum), True)

  def forward(self, dummy=None): # dummy is needed for lightning, maybe?
    if self.conditional_generation:
      return self.x
    else:
      return self.x, self.y

class Distiller3D_Dropout(nn.Module):
  def __init__(self, c, h, w, n_classes, batch_size, inner_lr=.02, inner_momentum=None, conditional_generation=False, dropout=0.5):
    super(Distiller3D_Dropout, self).__init__()
    self.conditional_generation = conditional_generation

    self.x = nn.Parameter(torch.randn((batch_size, c, h, w)), True)
    if not conditional_generation:

      self.y = nn.Parameter(torch.randn((batch_size, n_classes)), True)

    # Inner optimizer parameters
    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr), True)
    if inner_momentum is not None:
      self.inner_momentum = nn.Parameter(torch.tensor(inner_momentum), True)

    if dropout != 0:
      self.dropout = nn.Dropout(p=dropout)
    else:
      self.dropout = None

  def forward(self, dummy=None): # dummy is needed for lightning, maybe?
    if self.dropout:
      return self.dropout(self.x) if self.conditional_generation else (self.dropout(self.x), self.y)
    if self.conditional_generation:
      return self.x
    else:
      return self.x, self.y


# Distiller w/ 1D input (just # features) 
class Distiller1D(nn.Module):
  def __init__(self, batch_size, input_features, num_classes, inner_lr=.02, inner_momentum=None, conditional_generation=False):
    super(Distiller1D, self).__init__()
    self.conditional_generation = conditional_generation

    x = torch.randn((batch_size, input_features))
    self.x = nn.Parameter(x, True)
    if not conditional_generation:
      self.y = nn.Parameter(torch.randn((batch_size, num_classes)), True)

    # Inner optimizer parameters
    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr), True)
    if inner_momentum is not None:
      self.inner_momentum = nn.Parameter(torch.tensor(inner_momentum), True)

  def forward(self):
    if self.conditional_generation:
      return self.x
    else:
      return self.x, self.y

class SimpleConvNet(nn.Module):
  def __init__(self, c, h, w, n_classes):
    super(SimpleConvNet, self).__init__()
    self.convs = nn.Sequential(
      nn.Conv2d(c, 16, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, padding=1),
      nn.ReLU()
    )
    self.lin = nn.Linear(32*h*w, n_classes)

  def forward(self, x):
    x = self.convs(x)
    return self.lin(x.flatten(1))

class RandomArchConvNet(nn.Module):
  def __init__(self, c, h, w, n_classes):
    super(RandomArchConvNet, self).__init__()
    hs = [8, 16, 32, 64, 128]
    hid = hs[random.randint(0,len(hs)-1)]
    relu = nn.ReLU()
    l = random.randint(1,5) # number of random hidden conv layers
    convs = [nn.Conv2d(c, hid//2, 3, padding=1), relu]
    for i in range(l):
      convs.append(nn.Conv2d((hid//2 if i == 0 else hid), hid, 3, padding=1))
      convs.append(relu) 
    self.convs = nn.Sequential(*convs)
    self.lin = nn.Linear(hid*h*w, n_classes)

  def forward(self, x):
    x = self.convs(x)
    return self.lin(x.flatten(1))

class BayesConvNet(PyroModule):
  def __init__(self, c, h, w, n_classes, device):
    super(BayesConvNet, self).__init__()
    self.c1 = PyroModule[nn.Conv2d](c, 16, 3, padding=1)
    self.c2 = PyroModule[nn.Conv2d](16, 32, 3, padding=1)
    
    self.lin = PyroModule[nn.Linear](32*h*w, n_classes)

  
    # Define priors p(θ)
    # This is the only way I could find to put the module on the device: .to(device) doesn't work...
    self.c1.weight  = PyroSample(dist.Normal(torch.tensor(0.).to(device),10.).expand(self.c1.weight.shape).to_event(self.c1.weight.dim()))
    self.c1.bias    = PyroSample(dist.Normal(torch.tensor(0.).to(device),10.).expand(self.c1.bias.shape).to_event(self.c1.bias.dim()))
    self.c2.weight  = PyroSample(dist.Normal(torch.tensor(0.).to(device),10.).expand(self.c2.weight.shape).to_event(self.c2.weight.dim()))
    self.c2.bias    = PyroSample(dist.Normal(torch.tensor(0.).to(device),10.).expand(self.c2.bias.shape).to_event(self.c2.bias.dim()))
    self.lin.weight = PyroSample(dist.Normal(torch.tensor(0.).to(device),10.).expand(self.lin.weight.shape).to_event(self.lin.weight.dim()))
    self.lin.bias   = PyroSample(dist.Normal(torch.tensor(0.).to(device),10.).expand(self.lin.bias.shape).to_event(self.lin.bias.dim()))

  def forward(self, x, y=None):
    sigma = pyro.sample("sigma", dist.Uniform(0.,10.))
    x = F.relu(self.c1(x))
    x = F.relu(self.c2(x))
    mean = self.lin(x.flatten(1))
    # loss function defined in here:
    with pyro.plate('data', mean.shape[0]):
        print(sigma.shape)
        print(mean.shape)
        print(y.shape)
        obs = pyro.sample('obs', dist.Normal(mean, sigma), obs=y)
        print(obs.shape)
    return mean

class CartpoleActor(nn.Module):
  def __init__(self, state_size=4, action_size=2):
    super(CartpoleActor, self).__init__()

    hidden_size = 64

    # Note: Weight norm does not help Cartpole Distillation!!!
    self.net = nn.Sequential(rl_layer_init(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             rl_layer_init(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             rl_layer_init(nn.Linear(hidden_size, action_size), std=.01))


  def forward(self, x):
    return self.net(x.view(x.size(0),-1))

# Return a single value for a state, estimating the future discounted reward of following the current policy (it's tied to the PolicyNet it trained with)
class CartpoleCritic(nn.Module):
  def __init__(self, state_size=4):
    super(CartpoleCritic, self).__init__()

    hidden_size = 64

    self.net = nn.Sequential(rl_layer_init(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             rl_layer_init(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             rl_layer_init(nn.Linear(hidden_size, 1), std=1.))

  def forward(self, x):
    return self.net(x.view(x.size(0),-1))


class AtariActor(nn.Module): # an actor-critic neural network
    def __init__(self, state_channels, num_actions):
        super(AtariActor, self).__init__()

        self.convs = nn.Sequential(
          rl_layer_init(nn.Conv2d(state_channels, 32, 8, stride=4)),
          nn.ReLU(),
          rl_layer_init(nn.Conv2d(32,64, 4, stride=2)),
          nn.ReLU(),
          rl_layer_init(nn.Conv2d(64, 64, 3, stride=1)),
          nn.ReLU()
        )
        h, w = (7,7)
        self.head = nn.Sequential(
          rl_layer_init(nn.Linear(64*h*w, 512)),
          nn.ReLU(),
          rl_layer_init(nn.Linear(512, num_actions), std=.01)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class AtariCritic(nn.Module): # an actor-critic neural network
    def __init__(self, state_channels):
        super(AtariCritic, self).__init__()

        self.convs = nn.Sequential(
          rl_layer_init(nn.Conv2d(state_channels, 32, 8, stride=4)),
          nn.ReLU(),
          rl_layer_init(nn.Conv2d(32,64, 4, stride=2)),
          nn.ReLU(),
          rl_layer_init(nn.Conv2d(64, 64, 3, stride=1)),
          nn.ReLU()
        )
        h, w = (7,7)
        self.head = nn.Sequential(
          rl_layer_init(nn.Linear(64*h*w, 512)),
          nn.ReLU(),
          rl_layer_init(nn.Linear(512, 1), std=1.)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# INITIALIZATION FUNCTIONS #
ROOT_2 = 2**.5

def rl_layer_init(layer, std=ROOT_2, bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

# Dataset that wraps memory for a dataloader
class RLDataset(torch.utils.data.Dataset):
  def __init__(self, rollout):
    super().__init__()
    self.rollout = rollout
    self.width = len(rollout) # Number of distinct value types saved (state, action, etc.)
    rollout_shape = rollout[0].shape
    self.rollout_len, self.num_envs = rollout_shape[0], rollout_shape[1]
    self.full = self.rollout_len * self.num_envs

  def __getitem__(self, index):
    return [self.rollout[i][index//self.num_envs][index%self.num_envs] for i in range(self.width)]

  def __len__(self):
    return self.full