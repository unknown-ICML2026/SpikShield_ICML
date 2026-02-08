from ctypes import sizeof
from re import S
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from data_loader.datasets import Dataset
import torch
from torch.multiprocessing import Pool, Process, set_start_method
import pickle as pk
import random
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class BodyResponseDataset(Dataset):
  def __init__(self, path, classes=15, transform=None, prior_labels=None):
    df = pd.read_excel(path,engine='openpyxl')

    self.transform = transform

    df = df.set_index("index", drop=True)

    if prior_labels is not None:
      self.target = list(prior_labels)
    else:
      self.target = [2,3,4,5,6,7,8,11,12,13,14,17,18,19,20]
      if classes == 9:
        self.target = [3,4,5,6,8,12,14,17,18]

    df_ = pd.DataFrame()
    for target in self.target:
      df_ = pd.concat(
          [df_, df[df["Response"] == target]],
          ignore_index=True
      )
    df = df_

    self.dataSize = len(df.index)
    self.targetMap = np.zeros(21)
    for i, target in enumerate(self.target):
      self.targetMap[target] = i
    self.y  = [self.targetMap[i] for i in df["Response"].iloc[:]]

    x = df.drop("Response",axis=1)
    mean = x.mean()
    std  = x.std()
    x = (x - mean).div(std)

    self.x = [x.iloc[i].tolist() for i in range(self.dataSize)]

  def __len__(self):
    return self.dataSize

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    if isinstance(idx, list):
      size = len(idx)
    elif isinstance(idx, int):
      size = 1
    else:
      size = idx.size

    if size > 1:
      x, y = [self.x[i] for i in idx], [self.y[i] for i in idx]
    else:
      x, y = self.x[idx], self.y[idx]

    if self.transform:
      x = self.transform(x)
      y = self.transform(y)

    return x, y.clone().detach().long()

  def get_mean_processed(self):
    return torch.tensor(np.mean(self.x, axis=0), dtype=torch.float32)
