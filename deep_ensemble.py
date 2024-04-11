import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision

import torchmetrics

from models import Distiller3D, SimpleConvNet 
from helper import load_distiller_sd_fabric

import matplotlib.pyplot as plt

cal_error_f = torchmetrics.classification.MulticlassCalibrationError(10)

def evaluate(forward, loader):
  with torch.no_grad():
    loss = 0; n_correct = 0; cal = 0
    for x, y in loader:
      y_hat = forward(x)
      loss += F.cross_entropy(y_hat, y, reduction='sum').item()
      n_correct += (y_hat.argmax(1) == y).bool().sum().item()
      cal = cal_error_f(y_hat, y)*y_hat.size(0) # get sum instead of mean
  return loss, n_correct, cal

def ensemble_prediction(x):
  # TODO: Maybe try weighting by overall accuracy, accuracy on a given class, etc
  return torch.mean(torch.stack([model(x) for model in ensemble], dim=0), dim=0)

if __name__ == '__main__':
  mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)])
  train_dataset = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=mnist_transform, download=True)
  val_dataset   = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=False, transform=mnist_transform, download=False)
  c, h, w = train_dataset[0][0].shape
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
  val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=256, shuffle=True)

  # distiller_sd = load_distiller_sd_fabric('~/Documents/Distillation_Geometry/sl_exp/mnist_normalized_distill_4gpu_fixed/checkpoint2/state.ckpt')
  distiller_sd = load_distiller_sd_fabric('./mnist_normalized_distill_b256/checkpoint2/state.ckpt')
  distiller = Distiller3D(c, h, w, 10, 256)
  distiller.load_state_dict(distiller_sd)
  xd, yd = distiller()

  ensemble_size = 50
  
  # Inner learning to create ensembles
  ensemble = [SimpleConvNet(c, h, w, 10) for _ in range(ensemble_size)]
  for model in ensemble:
    inner_optim = optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
    y_hat = model(xd)
    F.mse_loss(y_hat, yd).backward()
    inner_optim.step()
    inner_optim.zero_grad()

  mean_losses = [[],[]]
  accs = [[],[]]
  cals = [[],[]]
  
  for i,(loader,len_data) in enumerate(((train_loader, len(train_dataset)), (val_loader, len(val_dataset)))):
    
    for model in ensemble:
      sum_loss, n_correct, cal = evaluate(model.forward, loader)
      mean_losses[i].append(sum_loss / len_data)
      accs[i].append(n_correct / len_data)
      cals[i].append(cal)
    sum_loss, n_correct, cal = evaluate(ensemble_prediction, loader)
    mean_losses[i].append(sum_loss / len_data)
    accs[i].append(n_correct / len_data)
    cals[i].append(cal)

  # todo: print table
  print("Training:\tMean\tBest\tEnsemble")
  print("Loss:\t{}\t{}\t{}".format(np.mean(mean_losses[0][:-1]), np.min(mean_losses[0][:-1]), mean_losses[0][-1]))
  print("Acc :\t{}\t{}\t{}".format(np.mean(accs[0][:-1]), np.max(accs[0][:-1]), accs[0][-1]))
  print("Cal :\t{}\t{}\t{}".format(np.mean(cals[0][:-1]), np.min(cals[0][:-1]), cals[0][-1]))
  
  print("Validation")
  print("Loss:\t{}\t{}\t{}".format(np.mean(mean_losses[1][:-1]), np.min(mean_losses[1][:-1]), mean_losses[1][-1]))
  print("Acc :\t{}\t{}\t{}".format(np.mean(accs[1][:-1]), np.max(accs[1][:-1]), accs[1][-1]))
  print("Cal :\t{}\t{}\t{}".format(np.mean(cals[1][:-1]), np.min(cals[1][:-1]), cals[1][-1]))



# TODO: Derive posterior predictive