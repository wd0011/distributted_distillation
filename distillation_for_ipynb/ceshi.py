import argparse
import torch
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


import time
from model import CNN_Model
Net = CNN_Model()

model_set = []
a = []
for worker_id in range(5):
    model_set.append(Net)
    a.append(CNN_Model())

print(model_set[0] is model_set[1])
print(a[0] is a[1])

