'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-05-25 22:29:47
@LastEditors: 吴宇辉
@LastEditTime: 2020-05-28 11:41:50
'''

import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import torchvision
import utils

learning_rate = 0.0001      # 学习率设置

root = os.getcwd() 

def default_loader(path): 
    return Image.open(path).convert('L') 

class PneumoniaDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(PneumoniaDataset, self).__init__()

        fh = open(txt, 'r') #按照传入的路径和txt文本参数，以只读的方式打开这个文本 
        imgs = []
        for line in fh: 
            #按照传入的路径和txt文本参数，以只读的方式打开这个文本
            line = line.strip('\n')
            line = line.rstrip('\n')    # 删除 本行string 字符串末尾的指定字符
            words = line.split()    #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))      #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定

        self.imgs = imgs 
        self.transform = transform 
        self.target_transform = target_transform 
        self.loader = loader 
        fh.close() 
    
    #使用__getitem__()对数据进行预处理并返回想要的信息
    def __getitem__(self, index):   #这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]    #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)   # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)   #数据标签转换为Tensor

        return img,label 
    
    #使用__len__()初始化一些需要传入的参数及数据集的调用
    def __len__(self):
        #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
    
def main():
    train_data = PneumoniaDataset(txt=root+'/txt/train.txt', transform=torchvision.transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor()])) 
    test_data = PneumoniaDataset(txt=root+'/txt/test.txt', transform=torchvision.transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor()]))

    train_loader = DataLoader(dataset=train_data, batch_size=5, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=5, shuffle=False, num_workers=4)

    classes = ('normal', 'bacteria', 'virus')

    # # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    # # show images
    # utils.imshow(torchvision.utils.make_grid(images))
    # #print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4) ))

    # x, y = next(iter(train_loader))
    # print(x.shape, y.shape, x.min(), y.min())

    net = torchvision.models.inception_v3(pretrained=True)

    criterion = nn.CrossEntropyLoss()  #交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 

    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            print(data)
            #输入数据
            #读取数据的数据内容和标签
            inputs, labels = data 
            print(inputs)
            print(labels)
            inputs, labels = Variable(inputs), Variable(labels)
            print(inputs)
            print(labels)

            #梯度清零，也就是把loss关于weight的导数变成0
            optimizer.zero_grad() 

            #forward + backward
            #得到网络的输出
            outputs = net(inputs) 

            #计算损失值，将输出的outputs和原来导入的labels作为loss函数的输入就可以得到损失了：
            loss = criterion(outputs, labels)  #output 和 labels的交叉熵损失
            #计算得到的loss后就要回转损失
            loss.backward()
            #loss.backward()，有时候，我们并不想所有variable的梯度。那就要
            #可以通过vatiable的两个参数
            #更新参数
            #回传损失过程会计算梯度，然后需要更具这些梯度更新参数，oprimizer.step()就是用来更新参数的。
            # oprimizer.step()后，你就可以从optimizer.param_groups[0]['params']里面看到各个层的地图和权值信息
            optimizer.step()    #利用计算的得到的梯度对参数进行更新

            #打印log信息
            running_loss += loss.item()     #用于从tensor中获取python数字
            if i % 2000 == 1999:    #每2000个batch打印一次训练状态
                print('[%d, %5d] loss: %.3f' % (epoch, i+1, running_loss /2000))

                running_loss = 0.0
    print('Finished Training')

if __name__ == "__main__":
    main()
        

