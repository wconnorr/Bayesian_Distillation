import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision

import torchmetrics

from models import Distiller3D, AtariActor, AtariCritic
from helper import load_distiller_sd_fabric

import matplotlib.pyplot as plt

import vector_env

from rl_helper import perform_episode, simple_act, ensemble_act

def ensemble_prediction(x):
  # TODO: Maybe try weighting by overall accuracy, accuracy on a given class, etc
  return torch.mean(torch.stack([model(x) for model in ensemble], dim=0), dim=0)

if __name__ == '__main__':
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  env = vector_env.make_atari("CentipedeNoFrameskip-v4")
  n_actions = env.action_space.n
  c, h, w = env.observation_space.shape

  distiller_sd = torch.load('./centipede_distiller_sd.pt', map_location=device)
  distiller = Distiller3D(c, h, w, n_actions, n_actions//2+1).to(device)
  distiller.load_state_dict(distiller_sd)
  xd, yd = distiller()

  ensemble_size = 25
  
  # Inner learning to create ensembles
  ensemble = [AtariActor(c, n_actions).to(device) for _ in range(ensemble_size)]
  for model in ensemble:
    inner_optim = optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
    y_hat = model(xd)
    F.mse_loss(y_hat, yd).backward()
    inner_optim.step()
    inner_optim.zero_grad()

  rewards = []
  for model in ensemble:
    rewards.append(perform_episode(model, simple_act, env, device))
  ensemble_reward = perform_episode(ensemble, ensemble_act, env, device)

  # todo: print table
  print("Mean reward = {}".format(np.mean(rewards)))
  print("Max  reward = {}".format(np.max(rewards)))
  print("Ens  reward = {}".format(ensemble_reward))