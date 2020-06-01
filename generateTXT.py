'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-05-27 17:20:45
@LastEditors: 吴宇辉
@LastEditTime: 2020-05-27 23:06:08
'''

import os
import re

def generateTXT(filePath, geneTXTname):
    '''fileName为输入文件名称，geneTXTname为生成txt格式文件名称'''
    fl = open(geneTXTname+'.txt', 'a+') 

    for i in os.walk( filePath ): 
        for item in i[2]:       #i[2]获得文件目录下所有文件名称构成的list
            if re.findall('bacteria', item):
                fl.write(filePath+'/'+item+' 1'+'\n')
            elif re.findall('virus', item):
                fl.write(filePath+'/'+item+' 2'+'\n')
            elif re.findall('jpeg', item):      #如果传入的是正常图片，给它赋值label = 0
                fl.write(filePath+'/'+item+' 0'+'\n')
            
    fl.close() 

def main():
    trainName = 'train' 
    testName = 'test' 
    valName = 'val' 
    trainPath = './chest_xray/train' 
    testPath = './chest_xray/test' 
    valPath = './chest_xray/val' 
    PNEUMONIA = 'PNEUMONIA' 
    NORMAL = 'NORMAL' 
    PNEUMONIAPATH = '/' + PNEUMONIA 
    NORMALPATH = '/' + NORMAL 
    
    # generateTXT(trainPath+NORMALPATH, trainName+NORMAL, 0) 
    # generateTXT(testPath+PNEUMONIAPATH, testName+PNEUMONIA)
    # generateTXT(testPath+NORMALPATH, testName+NORMAL, 0)
    # generateTXT(valPath+PNEUMONIAPATH, valName+PNEUMONIA)
    # generateTXT(valPath+NORMALPATH, valName+NORMAL, 0)
    # generateTXT(trainPath+PNEUMONIAPATH, trainName+PNEUMONIA)
    generateTXT(testPath+NORMALPATH, 'test')

if __name__ == "__main__":
    main() 

