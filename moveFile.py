'''
@Description: 
@Version: 2.0
@Autor: 吴宇辉
@Date: 2020-05-28 21:28:18
@LastEditors: 吴宇辉
@LastEditTime: 2020-05-28 21:55:58
'''
import shutil
import os
import re 

def main():
    '''移动文件，将bacteria图片和virus图片分开'''
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
    BACTERIA = '/BACTERIA'
    VIRUS = '/VIRUS'
    
    dirPath = 'D:\\code\\homework\\junior2\\cxsj\\project\\modelTest\\chest_xray\\val'
    filePath = dirPath + PNEUMONIAPATH
    BACTERIAPATH = dirPath+BACTERIA
    VIRUSPATH = dirPath+VIRUS

    for i in os.walk( filePath ): 
        for item in i[2]:       #i[2]获得文件目录下所有文件名称构成的list 
            if re.findall('bacteria', item): 
                shutil.copyfile(filePath+'/'+item, BACTERIAPATH+'/'+item) 
            elif re.findall('virus', item): 
                shutil.move(filePath+'/'+item, VIRUSPATH+'/'+item) 


if __name__ == "__main__":
    main()
