'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-06-03 14:24:42
@LastEditors: 吴宇辉
@LastEditTime: 2020-06-03 17:21:09
'''

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import itertools


initial_lr = 0.001

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

net_1 = model()

optimizer_1 = torch.optim.Adam(net_1.parameters(), lr = initial_lr)
scheduler_1 = StepLR(optimizer_1, step_size=20, gamma=0.5)

print("初始化的学习率：", optimizer_1.defaults['lr'])

for epoch in range(0, 50):
    # train

    optimizer_1.zero_grad()
    scheduler_1.step()
    optimizer_1.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    

