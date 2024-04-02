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

from laplace import Laplace

class OneBatchDataset(torch.utils.data.Dataset):
  def __init__(self, x, y):
    super().__init__()
    self.x = [xi for xi in x]
    self.y = [yi for yi in y]
  
  def __len__(self):
    return len(self.x)

  def __getitem__(self, i):
    return self.x[i], self.y[i]

cal_error_f = torchmetrics.classification.MulticlassCalibrationError(10)

def evaluate(model, loader, device):
  with torch.no_grad():
    loss = 0; n_correct = 0; cal = 0
    for x, y in tqdm(loader):
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

  distiller_sd = load_distiller_sd_fabric('~/Documents/Distillation_Geometry/sl_exp/mnist_normalized_distill_4gpu_fixed/checkpoint2/state.ckpt')
  distiller = Distiller3D(c, h, w, 10, 6).to(device)
  distiller.load_state_dict(distiller_sd)
  xd, yd = distiller()
  distiller_loader= torch.utils.data.DataLoader(OneBatchDataset(xd.detach().cpu(), yd.detach().cpu()), batch_size=xd.size(0), shuffle=False) # only 1 batch possible

  num_trials = 25

  mean_losses = []
  accs = []
  cals = []

  l_mean_losses = []
  l_accs = []
  l_cals = []
  
  # Inner learning 
  print("Learner type \tloss\tacc\tcal\n")
  for trial in range(num_trials):
    len_data = len(val_dataset)
    # model = SimpleConvNet(c, h, w, 10)
    # For whatever reason, it needs to be a default preexisting nn.Module???
    model = nn.Sequential(
      nn.Conv2d(c, 16, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, padding=1),
      nn.ReLU(),
      nn.Flatten(1),
      nn.Linear(32*h*w, 10)
    ).to(device)
    
    inner_optim = optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
    y_hat = model(xd)
    F.mse_loss(y_hat, yd).backward()
    inner_optim.step()
    inner_optim.zero_grad()

    # Validate standard learner
    sum_loss, n_correct, cal = evaluate(model, val_loader, device)
    mean_losses.append(sum_loss / len_data)
    accs.append(n_correct / len_data)
    cals.append(cal)
    print("Std Learner {}:\t{:.4f}\t{:.4f}\t{:.4f}".format(trial, mean_losses[-1], accs[-1], cals[-1]))

    # Laplace Approx
    la = Laplace(model, likelihood='regression', subset_of_weights='all', hessian_structure='kron')
    print("model laplacified")
    del model # remove model to allow room for fitting on GPU
    la.fit(distiller_loader)
    print("laplace fit to distiller")
    la.optimize_prior_precision(method='marglik', val_loader=val_loader)
    print("laplace optimized on val data")
    
    # print(la(xd))
    # print(xd.shape)
    # for tup in la(xd):
    #   print(tup.shape)

    # Validate Laplace
    sum_loss, n_correct, cal = evaluate(la, val_loader, device)
    l_mean_losses.append(sum_loss / len_data)
    l_accs.append(n_correct / len_data)
    l_cals.append(cal)
    print("Lap Learner {}:\t{:.4f}\t{:.4f}\t{:.4f}".format(trial, l_mean_losses[-1], l_accs[-1], l_cals[-1]))
    print()

    quit()
    # TODO: Can we graph the Laplace approx onto / in comparison to the loss landscape?

  
  print("Validation\n")
  print("Loss Std:\t{}\t{}".format(np.mean(mean_losses), np.min(mean_losses)))
  print("Loss Lap:\t{}\t{}".format(np.mean(l_mean_losses), np.min(l_mean_losses)))
  print()
  print("Acc  Std:\t{}\t{}".format(np.mean(accs), np.max(accs)))
  print("Acc  Lap:\t{}\t{}".format(np.mean(l_accs), np.max(l_accs)))
  print()
  print("Cal  Std:\t{}\t{}".format(np.mean(cals), np.min(cals)))
  print("Cal  Lap:\t{}\t{}".format(np.mean(l_cals), np.min(l_cals)))