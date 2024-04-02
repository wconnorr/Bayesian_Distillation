# Performing Bayesian deep learning on the distilled dataset


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision

import torchmetrics

from tqdm import tqdm

from models import Distiller3D, SimpleConvNet, BayesConvNet
from helper import load_distiller_sd_fabric

import matplotlib.pyplot as plt

cal_error_f = torchmetrics.classification.MulticlassCalibrationError(10)

def evaluate(model, loader, device):
  with torch.no_grad():
    loss = 0; n_correct = 0; cal = 0
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      if type(y_hat) == tuple:
        y_hat = y_hat[0]
      loss += F.cross_entropy(y_hat, y, reduction='sum').item()
      n_correct += (y_hat.argmax(1) == y).bool().sum().item()
      cal = cal_error_f(y_hat, y)*y_hat.size(0)
  return loss, n_correct, cal


if __name__ == '__main__':
  device = torch.device('cpu')#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)])
  train_dataset = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=mnist_transform, download=True)
  val_dataset   = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=False, transform=mnist_transform, download=False)
  vallen = len(val_dataset)
  c, h, w = train_dataset[0][0].shape
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
  val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=256, shuffle=True)


  distiller_sd = load_distiller_sd_fabric('~/Documents/Distillation_Geometry/sl_exp/mnist_normalized_distill_4gpu_fixed/checkpoint2/state.ckpt')
  distiller = Distiller3D(c, h, w, 10, 6).to(device)
  distiller.load_state_dict(distiller_sd)
  xd, yd = distiller()

  num_trials = 25
  losses = []
  accs  = []
  cals = []
  
  for trial in tqdm(range(num_trials)):
    model = BayesConvNet(c, h, w, 10, device)
    
    # Inner learning
    guide = autoguide.AutoDiagonalNormal(model)
    inner_optim = pyro.optim.SGD({'lr': distiller.inner_lr.item()})
    svi = SVI(model, guide, inner_optim, loss=Trace_ELBO())
    model.train()
    loss = svi.step(xd, yd)
    

    # Validate standard learner
    loss, acc, cal = evaluate(model, val_loader, device)
    losses.append(loss / vallen)
    accs.append(acc / vallen)
    cals.append(cal)


print("Validation")
print("\tMEAN\tOPTIM")
print("Loss:\t{}\t{}".format(np.mean(losses), np.min(losses)))
print("Acc :\t{}\t{}".format(np.mean(accs) np.max(accs)))
print("Cal :\t{}\t{}".format(np.mean(cals), np.min(cals)))