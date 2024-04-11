# How does distillation affect calibration on the training and validation task?

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision

import torchmetrics

from tqdm import tqdm

from models import Distiller3D, SimpleConvNet 
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
  c, h, w = train_dataset[0][0].shape
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
  val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=256, shuffle=True)

  val_std = False
  val_distill = True

  if val_std:
    # Train
    num_trials = 25
    cals_t0_train = []
    cals_t0_val  = []
    cals_t0_valacc = []
    
    # Inner learning 
    for trial in tqdm(range(num_trials)):
      model = SimpleConvNet(c, h, w, 10).to(device)

      # LEARNING
      optimizer = optim.SGD(model.parameters(), lr=1e-3)
      for _ in range(10):
        for x, y in train_loader:
          x, y = x.to(device), y.to(device)
          y_hat = model(x)
          loss = F.cross_entropy(y_hat, y)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      
      # Validate standard learner
      cals_t0_train.append(evaluate(model, train_loader, device)[2])
      _, acc, cal = evaluate(model, val_loader, device)
      cals_t0_val.append(cal)
      cals_t0_valacc.append(acc/len(val_dataset))
    print("MNIST-Trained Learner: (mean acc={:.2f}%)\n\tTraining Calibration Mean = {:.4f}\n\tTraining Calibration Std = {:.4f}\n\tValidation Calibration Mean = {:.4f}\n\tValidation Calibration Std = {:.4f}".format(np.mean(cals_t0_valacc)*100, np.mean(cals_t0_train), np.std(cals_t0_train), np.mean(cals_t0_val), np.std(cals_t0_val)))

  if val_distill:
    # distiller_sd = load_distiller_sd_fabric('~/Documents/Distillation_Geometry/sl_exp/mnist_normalized_distill_4gpu_fixed/checkpoint2/state.ckpt')
    distiller_sd = load_distiller_sd_fabric('./mnist_normalized_distill_b256/checkpoint2/state.ckpt')
    distiller = Distiller3D(c, h, w, 10, 256).to(device)
    distiller.load_state_dict(distiller_sd)
    xd, yd = distiller()
  
    num_trials = 25
    cals_td_train = []
    cals_td_val  = []
    cals_td_valacc = []
    cals_td_valloss = []
    
    # Inner learning 
    for trial in tqdm(range(num_trials)):
      model = SimpleConvNet(c, h, w, 10).to(device)
      
      inner_optim = optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
      y_hat = model(xd)
      F.mse_loss(y_hat, yd).backward()
      inner_optim.step()
      inner_optim.zero_grad()
  
      # Validate standard learner
      cals_td_train.append(evaluate(model, train_loader, device)[2])
      loss, acc, cal = evaluate(model, val_loader, device)
      cals_td_val.append(cal)
      cals_td_valacc.append(acc/len(val_dataset))
      cals_td_valloss.append(loss/len(val_dataset))
    print("Distill-Trained Learner: (mean acc={:.2f}%,mean loss={:.4f})\n\tTraining Calibration Mean = {:.4f}\n\tTraining Calibration Std = {:.4f}\n\tValidation Calibration Mean = {:.4f}\n\tValidation Calibration Std = {:.4f}".format(np.mean(cals_td_valacc)*100, np.mean(cals_td_valloss), np.mean(cals_td_train), np.std(cals_td_train), np.mean(cals_td_val), np.std(cals_td_val)))