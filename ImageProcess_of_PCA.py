import numpy as np
import cv2 as cv
import os


# 数据中心化
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal

# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    k = Percentage2n(D, p)  # 确定k值
    print("保留99%信息，降维后的特征个数：" + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector


# 得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat


# PCA算法
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    # covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    # 得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)
    # 得到降维后的数据
    lowDataMat = getlowDataMat(dataMat, V)
    # 重构数据
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat


def IMGPCA(imagePath):
    image = cv.imread(imagePath)    # 路径上不能有中文，否则会报错找不到文件
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    rows, cols = image.shape

    reconImage = PCA(image, 0.99)
    reconImage = reconImage.astype(np.uint8)

    return reconImage # 返回灰度矩阵

# 使用PCA对数据图像进行处理
def image_PCA(n,image_base_path):

    result = np.array([])  # 创建一个空的一维数组
    print("开始使用PCA处理图像")
    for i in os.listdir(image_base_path):
        image = IMGPCA(image_base_path + '/' + i)
        image_arr_temp = np.array(image).reshape(16384) / 255

        # 行拼接，最终结果：共n行，一行3072列
        result = np.concatenate((result, image_arr_temp))

    # 这里reshape的参数 = 三通道数组长度之和
    result = result.reshape((n, 16384))

    return result

def MidBlock_PCA(matrix):
    mid = PCA(matrix, 0.99)
    mid = mid.astype(np.uint8)
    return mid