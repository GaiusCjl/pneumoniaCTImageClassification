'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-06-10 15:00:26
@LastEditors: 吴宇辉
@LastEditTime: 2020-06-13 10:51:42
'''
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

def plotloss(losses, accs):
    x_loss = range(len(losses['train']))
    plt.plot(x_loss, losses['train'], '-', label='train')
    plt.plot(x_loss, losses['test'], '-', label='valid')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(dir+'loss.png', dpi=1080)    
    plt.show()
    
    plt.plot(x_loss, accs['train'], '-', label='train')
    plt.plot(x_loss, accs['test'], '-', label='valid')
    plt.xlabel('epoches')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(dir+'acc.png', dpi=1080)    
    plt.show()

dir = 'Adam_150\\'
data = pd.read_csv(dir + 'result.txt', header=None)
print(data[0] )

losses = {'train': [i for ix, i in enumerate(data[0]) if ix % 2 == 0], 'test': [i for ix, i in enumerate(data[0]) if ix % 2 == 1]}
accs = {'train': [i for ix, i in enumerate(data[1]) if ix % 2 == 0], 'test': [i for ix, i in enumerate(data[1]) if ix % 2 == 1]}

plotloss(losses, accs)