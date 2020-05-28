'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-05-26 23:14:30
@LastEditors: 吴宇辉
@LastEditTime: 2020-05-27 16:10:43
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np 

transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
