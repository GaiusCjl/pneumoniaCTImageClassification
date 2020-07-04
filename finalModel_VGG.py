'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-06-18 10:37:56
@LastEditors: 吴宇辉
@LastEditTime: 2020-06-18 21:20:32
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
import matplotlib.pyplot as plt
import time
import os
import copy

from PIL import Image
import torchvision.transforms as T
# from models.mobilenet_master2 import MobileNet#导入自己定义的网络模型
from torch.autograd import Variable as V
import torch as t


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def main():
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "vgg"

    # Number of classes in the dataset
    num_classes = 3

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # 路径配置
    model_dir = 'D:/Users/19988/Desktop/vgg16_1.pkl'
    test_dir = 'D:/Users/19988/Desktop/hughshu/xray/chest_xray3/test'
    testData = test_dir + '/VIRUS/person44_virus_93.jpeg'

    #加载预训练模型
    if os.path.exists(model_dir):
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Send the model to GPU
        model_ft = model_ft.to(device)

        # model_ft.load_state_dict(torch.load(model_dir, map_location='cpu'))#加载训练好的模型文件
        model_ft = torch.load(model_dir, map_location='cpu')
        print('load ', model_dir, ' model parameters.\n')
    else:
        print("No such model, exiting...")
        exit()


    # Print the model we just instantiated
    print(model_ft)
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    

    # trans=T.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    #     ])
 
    #读入图片
    img = open(testData, 'rb')
    img = Image.open(img)  
    img = img.convert('RGB')
    
    img=data_transforms(img)    #这里经过转换后输出的img格式是[C,H,W],网络输入还需要增加一维批量大小B
    
    print(img.shape)
    img = img.unsqueeze(0)  #增加一维，输出的img格式为[1,C,H,W]
    print(img.shape)
    
    model = model_ft #导入网络模型
    model.eval()
    
    
    img = V(img)
    print(img)
    score = model(img) #将图片输入网络得到输出
    probability = torch.nn.functional.softmax(score,dim=1) #计算softmax，即该图片属于各类的概率
    max_value,index = torch.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别
    print(index)
    print(probability)

if __name__ == "__main__":
    main()
    