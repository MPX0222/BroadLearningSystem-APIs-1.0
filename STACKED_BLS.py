import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time

from BLS import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from ImageProcess_of_PCA import MidBlock_PCA


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

'''
参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长

N-----堆叠块数
th----临界值
'''
def StackBLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, N):

    print("the  1  floor of Stack BLS")
    train_u, test_u = BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3)
    endtrain = train_u
    endtest = test_u

    finaltrainacc = show_accuracy(endtrain, train_y)
    finaltestacc = show_accuracy(endtest, test_y)
    print('\nFinal Train accurate is', finaltrainacc * 100, '%')
    print('Final Testing accurate is', finaltestacc * 100, '%')

    for i in range(N - 1):
        train_yy = train_y - train_u
        test_yy = test_y - test_u
        print("\nthe ", i + 2, " floor of Stack BLS")
        train_u, test_u = BLS(train_u, train_yy, test_u, test_yy, s, c, 10, 10, 500)  # 暂时不知道这个参数和之前的存在什么关系
        endtrain = endtrain + train_u
        endtest = endtest + test_u

        finaltrainacc = show_accuracy(endtrain, train_y)
        finaltestacc = show_accuracy(endtest, test_y)
        print('\nFinal Train accurate is', finaltrainacc * 100, '%')
        print('Final Testing accurate is', finaltestacc * 100, '%')


# 存在问题
def StackBLS_AddEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M, N):
    print("the  1  floor of Stack BLS")
    train_u, test_u = BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3)
    endtrain = train_u
    endtest = test_u

    for i in range(N - 1):
        train_y = train_y - train_u
        test_y = test_y - test_u
        print("\nthe ", i + 2, " floor of Stack BLS")
        train_u, test_u = Stacked_BLS_AE(train_u, train_y, test_u, test_y, endtrain, endtest, s, c, 1, 1, 400, L, M)  # 暂时不知道这个参数和之前的存在什么关系
        endtrain = endtrain + train_u
        endtest = endtest + test_u

    finaltrainacc = show_accuracy(endtrain, train_y)
    finaltestacc = show_accuracy(endtest, test_y)
    print('\nFinal Train accurate is', finaltrainacc * 100, '%')
    print('Final Testing accurate is', finaltestacc * 100, '%')

# 存在问题
def StackBLS_AddFeatureEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M1, M2, M3, N):
    print("the  1  floor of Stack BLS")
    train_u, test_u = BLS_AddFeatureEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M1, M2, M3)
    endtrain = train_u
    endtest = test_u

    for i in range(N - 1):
        train_y = train_y - train_u
        test_y = test_y - test_u
        print("\nthe ", i + 2, " floor of Stack BLS")
        train_u, test_u = BLS_AddFeatureEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M1, M2, M3)  # 暂时不知道这个参数和之前的存在什么关系
        endtrain = endtrain + train_u
        endtest = endtest + test_u

    finaltrainacc = show_accuracy(endtrain, train_y)
    finaltestacc = show_accuracy(endtest, test_y)
    print('\nFinal Train accurate is', finaltrainacc * 100, '%')
    print('Final Testing accurate is', finaltestacc * 100, '%')

def Stacked_BLS_AE(train_x, train_y, test_x, test_y, train_u, test_u, s, c, N1, N2, N3, L, M):
    # 生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x, axis=1)  # 处理数据
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])

    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i + u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = pinvOfInput.dot(train_y)
    time_end = time.time()
    trainTime = time_end - time_start

    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain+train_u, train_y+train_u)
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (outputOfEachWindowTest - minOfEachWindow[i]) / \
                                                                  distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest+test_u, test_y+test_u)
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start = time.time()
        if N1 * N2 >= M:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M) - 1)
        else:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M).T - 1).T

        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s / np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd * parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer, OutputOfEnhanceLayerAdd])

        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e + 1] = Training_time
        OutputOfTrain = InputOfOutputLayer.dot(OutputWeightEnd)
        TrainingAccuracy = show_accuracy(OutputOfTrain+train_u, train_y+train_u)
        train_acc[0][e + 1] = TrainingAccuracy

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(
            InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest+test_u, test_y+test_u)

        Test_time = time.time() - time_start
        test_time[0][e + 1] = Test_time
        test_acc[0][e + 1] = TestingAcc
    print(train_acc)
    print(test_acc)

    return OutputOfTrain,OutputOfTest

# 尝试每层采取不同的处理方法
def MixedBlock_StackedBLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, N):

    print("the  1  floor of MixedBlock Stack BLS")
    train_u, test_u = BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3)
    endtrain = train_u
    endtest = test_u

    for i in range(N - 1):
        train_y = train_y - train_u
        test_y = test_y - test_u
        train_u = MidBlock_PCA(train_u)
        test_u = MidBlock_PCA(test_u)
        print("\nthe ", i + 2, " floor of MixedBlock Stack BLS")
        train_u, test_u = BLS(train_u, train_y, test_u, test_y, s, c, 1, 1, 400)  # 暂时不知道这个参数和之前的存在什么关系
        endtrain = endtrain + train_u
        endtest = endtest + test_u

    finaltrainacc = show_accuracy(endtrain, train_y)
    finaltestacc = show_accuracy(endtest, test_y)
    print('\nFinal Train accurate is', finaltrainacc * 100, '%')
    print('Final Testing accurate is', finaltestacc * 100, '%')



