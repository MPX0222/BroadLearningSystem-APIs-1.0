# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:35:24 2018


@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""

'''

Installation has been tested with Python 3.5.
Since the package is written in python 3.5, 
python 3.5 with the pip tool must be installed first. 
It uses the following dependencies: numpy(1.16.3), scipy(1.2.1), keras(2.2.0), sklearn(0.20.3)  
You can install these packages first, by the following commands:

pip install numpy
pip install scipy
pip install keras (if use keras data_load())
pip install scikit-learn
'''
import os
import numpy as np
import pandas as pd
import scipy.io as scio
import gzip
import sklearn

from BLS import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from BLS_LRF import BLS_LRF

from ImageProcess_of_Base import image_to_matrix_rgb, label_to_matrix, image_to_matrix_gray, rgb_tran_gray
from ImageProcess_of_CNN import image_CNN
from ImageProcess_of_PCA import image_PCA, PCA

''' 1.Keras Dataset (fashion_mnist/cifar10/cifar100) '''
'''
import keras
(traindata, trainlabel), (testdata, testlabel) = keras.datasets.mnist.load_data()
traindata = traindata.reshape(traindata.shape[0], 28*28).astype('float64')/255
trainlabel = keras.utils.to_categorical(trainlabel, 10)
testdata = testdata.reshape(testdata.shape[0], 28*28).astype('float64')/255
testlabel = keras.utils.to_categorical(testlabel, 10)
'''

''' 2.Initial Dataset(mnist) '''

'''
dataFile = './mnist.mat'
data = scio.loadmat(dataFile)

traindata = traindata / 255.0
testdata = testdata / 255.0

traindata = np.double(data['train_x']/255)
trainlabel = np.double(data['train_y'])
testdata = np.double(data['test_x']/255)
testlabel = np.double(data['test_y'])
'''

''' 3.New Dataset (VOC2007、Asian Giant)'''

traindir = 'D:\\ResearchOf_AI\\DataBase\\TargetDetectionSystem\\BLS Method\\train_data_rgb'
testdir = 'D:\\ResearchOf_AI\\DataBase\\TargetDetectionSystem\\BLS Method\\test_data_rgb'

'''
traindata = image_CNN(1000, traindir)
testdata = image_CNN(300, testdir)
trainlabel = label_to_matrix(1000,2,traindir)
testlabel = label_to_matrix(300,2,testdir)

scio.savemat('./DataSet/SHIP_PROCESS_VGG16RGB_traindata.mat', mdict={'traindata': traindata})
scio.savemat('./DataSet/SHIP_PROCESS_VGG16RGB_testdata.mat', mdict={'testdata': testdata})
scio.savemat('./DataSet/SHIP_PROCESS_VGG16RGB_trainlabel.mat', mdict={'trainlabel': trainlabel})
scio.savemat('./DataSet/SHIP_PROCESS_VGG16RGB_testlabel.mat', mdict={'trainlabel': testlabel})
'''

traindatamat = scio.loadmat('./DataSet/SHIP_PROCESS_VGG16GRAY_traindata.mat')
testdatamat = scio.loadmat('./DataSet/SHIP_PROCESS_VGG16GRAY_testdata.mat')
trainlabelmat = scio.loadmat('./DataSet/SHIP_PROCESS_VGG16GRAY_trainlabel.mat')
testlabelmat = scio.loadmat('./DataSet/SHIP_PROCESS_VGG16GRAY_testlabel.mat')

traindata = np.double(traindatamat['traindata'])
testdata = np.double(testdatamat['testdata'])
trainlabel = np.double(trainlabelmat['trainlabel'])
testlabel = np.double(testlabelmat['trainlabel'])


''' 4.New Frame (BLS_LRF)'''
'''
traindir = 'D:\\ResearchOf_AI\\DataBase\\TargetDetectionSystem\\BLS Method\\train_data_rgb'
testdir = 'D:\\ResearchOf_AI\\DataBase\\TargetDetectionSystem\\BLS Method\\test_data_rgb'
'''

N1 = 20  #  # of nodes belong to each window
N2 = 20  #  # of windows -------Feature mapping layer
N3 = 1000 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps
M1 = 100  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient


print('-------------------BLS_BASE---------------------------')#
BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)

#print('-------------------STACK_BLS_BASE------------------------')
#StackBLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, 3)
#MixedBlock_StackedBLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, 3)

#print('-------------------BLS_ENHANCE------------------------')
#BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)

#print('-------------------STACK_BLS_ENHANCE------------------------')
#stackBLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, 3)

#M2 = 90  #  # of adding feature mapping nodes
#M3 = 90  #  # of adding enhance nodes

#print('-------------------BLS_FEATURE&ENHANCE----------------')
#BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

#print('-------------------STACK_BLS_FEATURE&ENHANCE------------------------')
#stackBLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3, 5)

#print('-------------------BLS_LRF------------------------')
#BLS_LRF(1000, 300, traindir,testdir, s, C, N1, N2, N3)

