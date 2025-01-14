# Testing distiller trained w/ MC Dropout

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision

import torchmetrics

from models import Distiller3D_Dropout, SimpleConvNet 
from helper import load_distiller_sd_fabric

import matplotlib.pyplot as plt

import cv2

import time

cal_error_f = torchmetrics.classification.MulticlassCalibrationError(10)

def evaluate(model, loader, device):
  with torch.no_grad():
    loss = 0; n_correct = 0; cal = 0
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      loss += F.cross_entropy(y_hat, y, reduction='sum').item()
      n_correct += (y_hat.argmax(1) == y).bool().sum().item()
      cal = cal_error_f(y_hat, y).item()*y_hat.size(0) # get sum instead of mean
  return loss, n_correct, cal

def ensemble_prediction(x):
  # TODO: Maybe try weighting by overall accuracy, accuracy on a given class, etc
  return torch.mean(torch.stack([model(x) for model in ensemble], dim=0), dim=0)

if __name__ == '__main__':
  start = time.time()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)])
  train_dataset = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=mnist_transform, download=True)
  val_dataset   = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=False, transform=mnist_transform, download=False)
  c, h, w = train_dataset[0][0].shape
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
  val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=256, shuffle=True)

  with torch.no_grad():
    distiller_sd = load_distiller_sd_fabric('./mnist_dropout_distill_b256/checkpoint2/state.ckpt')
    distiller = Distiller3D_Dropout(c, h, w, 10, 256).to(device)
    distiller.load_state_dict(distiller_sd)
    # xd = 0
    # _, yd = distiller()
    # S = 6
    # print(yd[0])
    # for s in range(S):
    #   xdi, _ = distiller()
    #   cv2.imwrite('./mcd{}.png'.format(s), cv2.resize(xdi[0].squeeze(0).cpu().numpy()*256, (h*4, w*4), interpolation=cv2.INTER_NEAREST))
    #   xd += xdi
    # xd /= S
    # cv2.imwrite('./mcd_final.png'.format(s), cv2.resize(xd[0].squeeze(0).cpu().numpy()*256, (h*4, w*4), interpolation=cv2.INTER_NEAREST))
    # quit()
  
  num_models = 25

  mean_losses = []
  mean_accs = []
  mean_cals = []
  std_losses = []
  std_accs = []
  std_cals = []

  # STD Dropout Test
  distiller.eval()
  losses = []
  accs = []
  cals = []
  for _ in range(num_models):
    model = SimpleConvNet(c, h, w, 10).to(device)
    xd, yd = distiller()
    
    inner_optim = optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
    y_hat = model(xd)
    F.mse_loss(y_hat, yd).backward()
    inner_optim.step()
    inner_optim.zero_grad()
  
    len_data = len(val_dataset)
    sum_loss, n_correct, cal = evaluate(model, val_loader, device)
    losses.append(sum_loss / len_data)
    accs.append(n_correct / len_data)
    cals.append(cal)
  
  print("Validation STD Dropout")
  d_loss = np.mean(losses)
  print("Loss:\t{}\t{}\t{}".format(d_loss, np.std(losses), np.min(losses)))
  d_acc = np.mean(accs)
  print("Acc :\t{}\t{}\t{}".format(d_acc, np.std(accs), np.max(accs)))
  d_cal = np.mean(cals)
  print("Cal :\t{}\t{}\t{}".format(d_cal, np.std(cals), np.min(cals)))

  distiller.train()
  
  for S in range(1,101):
    losses = []
    accs = []
    cals = []
    for _ in range(num_models):
      model = SimpleConvNet(c, h, w, 10).to(device)
      xd = 0
      _, yd = distiller()
      for s in range(S):
        xdi, _ = distiller()
        xd += xdi
      xd /= S
      
      inner_optim = optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
      y_hat = model(xd)
      F.mse_loss(y_hat, yd).backward()
      inner_optim.step()
      inner_optim.zero_grad()
    
      len_data = len(val_dataset)
      sum_loss, n_correct, cal = evaluate(model, val_loader, device)
      losses.append(sum_loss / len_data)
      accs.append(n_correct / len_data)
      cals.append(cal)
    
    print("Validation: S={}".format(S))
    print("Loss:\t{}\t{}\t{}".format(np.mean(losses), np.std(losses), np.min(losses)))
    print("Acc :\t{}\t{}\t{}".format(np.mean(accs), np.std(accs), np.max(accs)))
    print("Cal :\t{}\t{}\t{}".format(np.mean(cals), np.std(cals), np.min(cals)))

    mean_losses.append(np.mean(losses))
    std_losses.append(np.std(losses))
    mean_accs.append(np.mean(accs))
    std_accs.append(np.std(accs))
    mean_cals.append(np.mean(cals))
    std_cals.append(np.std(cals))

  fig = plt.figure()
  xs = list(range(1,101))
  plt.errorbar(xs, mean_losses, std_losses, label='Monte Carlo Dropout')
  plt.plot([1,51], [d_loss]*2, color='black', label='Standard Dropout')
  plt.title("MC Dropout Validation")
  plt.ylabel("Loss")
  plt.xlabel("S")
  fig.savefig("mcd_loss_b256_3.png", dpi=fig.dpi)
  plt.close('all')
  
  fig = plt.figure()
  plt.errorbar(xs, mean_accs, std_accs, label='Monte Carlo Dropout')
  plt.plot([1,51], [d_acc]*2, color='black', label='Standard Dropout')
  plt.title("MC Dropout Validation")
  plt.ylabel("Accuracy")
  plt.xlabel("S")
  fig.savefig("mcd_accs_b256_3.png", dpi=fig.dpi)
  plt.close('all')
  
  fig = plt.figure()
  plt.errorbar(xs, mean_cals, std_cals, label='Monte Carlo Dropout')
  plt.plot([1,51], [d_cal]*2, color='black', label='Standard Dropout')
  plt.title("MC Dropout Validation")
  plt.ylabel("Calibration")
  plt.xlabel("S")
  fig.savefig("mcd_cals_b256_3.png", dpi=fig.dpi)
  plt.close('all')
  
  print(time.time() - start)