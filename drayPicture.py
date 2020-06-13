'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-06-10 14:39:51
@LastEditors: 吴宇辉
@LastEditTime: 2020-06-10 14:39:59
'''

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import visdom
import matplotlib.pyplot as plt
import time
import os
import copy
import utils

def get


def drawPic():
    x_loss = range(len(losses['train']))
    plt.plot(x_loss, losses['train'], '-', label='train')
    plt.plot(x_loss, losses['test'], '-', label='valid')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png', dpi=1080)    
    plt.show()
        
    plt.plot(x_loss, accs['train'], '-', label='train')
    plt.plot(x_loss, accs['test'], '-', label='valid')
    plt.xlabel('epoches')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('acc.png', dpi=1080)    
    plt.show() 

def main():


if __name__ == "__main__":
    main()
