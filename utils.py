'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-05-27 23:40:58
@LastEditors: 吴宇辉
@LastEditTime: 2020-05-27 23:44:55
'''

import matplotlib.pyplot as plt 
import numpy as np 

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


