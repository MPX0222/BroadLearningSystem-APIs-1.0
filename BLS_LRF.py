import os
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time
import scipy.io as scio

from ImageProcess_of_Base import image_to_matrix_rgb, label_to_matrix
from BLS import BLS
from ImageProcess_of_CNN import image_CNN, CNN_Model


def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count / len(Label), 5))


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

def BLS_LRF(n1, n2, train_path, test_path, s, C, N1, N2, N3):

    time_start = time.time()  # 计时开始

    traindata = image_to_matrix_rgb(n1, train_path)
    testdata = image_to_matrix_rgb(n2, test_path)
    trainlabel = label_to_matrix(n1, 2, train_path)
    testlabel = label_to_matrix(n2, 2, test_path)

    StackofBLS_train, StackofBLS_test = BLS_re(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
    StackofLRF_train = image_CNN(n1, train_path)
    StackofLRF_test = image_CNN(n2, test_path)

    StackofTrain = np.hstack([StackofBLS_train, StackofLRF_train])
    StackofTest = np.hstack([StackofBLS_test, StackofLRF_test])

    time_end = time.time()  # 计时结束
    trainTime = time_end - time_start

    pinvOfInput = pinv(StackofTrain, C)
    # 输出权重 = 标签 乘以 伪逆
    OutputWeight = np.dot(pinvOfInput, trainlabel)
    time_end = time.time()  # 计时结束
    trainTime = time_end - time_start

    # 训练预测输出 = 最终输入 乘以 输出权重
    OutputOfTrain = np.dot(StackofTrain, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, trainlabel)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')

    time_start = time.time()
    OutputOfTest = np.dot(StackofTest, OutputWeight)
    time_end = time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, testlabel)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')

    scio.savemat('./DataSet/SHIP_PROCESS_BLSLRF_traindata.mat', mdict={'traindata': StackofTrain})
    scio.savemat('./DataSet/SHIP_PROCESS_BLSLRF_testdata.mat', mdict={'testdata': StackofTest})
    scio.savemat('./DataSet/SHIP_PROCESS_BLSLRF_trainlabel.mat', mdict={'trainlabel': trainlabel})
    scio.savemat('./DataSet/SHIP_PROCESS_BLSLRF_testlabel.mat', mdict={'trainlabel': testlabel})

    return 0




def BLS_re(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):

    L = 0
    # 预处理数据，axis等于1时标准化每个样本（行）,axis等于0时独立地标准化每个特征
    train_x = preprocessing.scale(train_x, axis=1)
    # 将输入矩阵进行行链接，即平铺展开整个矩阵
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1

    for i in range(N2):
        random.seed(i)
        # 随机化权重
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        # 源输入X的平铺展开矩阵与随机化权重W相乘 = 每个窗口的特征结点
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        # 对上述结果归一化处理
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        # 进行标准化
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        # 随机化贝塔值
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        # 源输入X的平铺展开矩阵与贝塔值相乘 = 每个窗口的输出
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        # 求解输出最大值与最小值的距离
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]

        # 生成映射节点最终输入
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        # del outputOfEachWindow
        # del FeatureOfEachWindow
        # del weightOfEachWindow

    # 生成增强结点
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    # 增强结点乘上相应随机权重
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    # 参数的归一化？
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    # 生成增强节点的最终输入
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])

    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()

    # 按照上述方法处理测试集数据，然后将 测试集最终输入 乘以 输出权重，再进行精度比较
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin

    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    # 测试集最终输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

    return InputOfOutputLayer, InputOfOutputLayerTest