import torch
import numpy as np
from lightning import Fabric

def load_distiller_sd_fabric(filepath):
  fabric = Fabric()
  state = fabric.load(filepath)
  return state['distiller']