from BLS import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes

import os
import torch

import torchvision.transforms as transforms
import torch.nn as nn
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import time
import matplotlib.pyplot as plt


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and testing datasets.
pic_dir = 'D:\\CodeWorkshop\\test_data_rgb\\1045_1.jpg'

# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()


def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    #img256 = skimage.transform.resize(img, (256, 256))
    img256 = np.asarray(img)
    img256 = img256.astype(np.float32)

    return transform(img256)



def CNN_Model(input):

    conv1 = nn.Sequential(
        nn.Conv2d(3, 32, 5, 1, 2),  # input_size=(3*256*256)，padding=2
        nn.ReLU(),  # input_size=(32*256*256)
        nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(32*128*128)
    )

    # 第二层神经网络，包括卷积层、线性激活函数、池化层
    conv2 = nn.Sequential(
        nn.Conv2d(32, 64, 5, 1, 2),  # input_size=(32*128*128)
        nn.ReLU(),  # input_size=(64*128*128)
        nn.MaxPool2d(2, 2)  # output_size=(64*64*64)
    )

    conv3 = nn.Sequential(
        nn.Conv2d(64, 96, 5, 1, 2), # input_size=(64*64*64)
        nn.ReLU(),  # input_size=(96*64*64)
        nn.MaxPool2d(2,2)   # output_size=(96*32*32)
    )

    # 全连接层(将神经网络的神经元的多维输出转化为一维)
    fc1 = nn.Sequential(
        nn.Linear(96 * 32 * 32, 128),  # 进行线性变换
        nn.ReLU()  # 进行ReLu激活
    )

    # 输出层(将全连接层的一维输出进行处理)
    fc2 = nn.Sequential(
        nn.Linear(128, 96),
        nn.ReLU()
    )

    VGG16_Conv1 = nn.Sequential(

        nn.Conv2d(3,64,kernel_size=3, padding=1), # inputsize (128,128,3)
        nn.ReLU(),

        nn.Conv2d(64,64,kernel_size=3,padding=1), # (128,128,64)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2), # (64,64,64)
    )

    VGG16_Conv2 = nn.Sequential(

        nn.Conv2d(64,128,kernel_size=3,padding=1),
        nn.ReLU(),

        nn.Conv2d(128,128,kernel_size=3,padding=1), # (64,64,128)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2), # (32,32,128)
    )

    VGG16_Conv3 = nn.Sequential(

        nn.Conv2d(128,256,kernel_size=3,padding=1),
        nn.ReLU(),

        nn.Conv2d(256,256,kernel_size=3,padding=1),
        nn.ReLU(),

        nn.Conv2d(256,256,kernel_size=3,padding=1), # (32,32,256)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2), # (16,16,256)
    )

    VGG16_Conv4 = nn.Sequential(

        nn.Conv2d(256,512,kernel_size=3,padding=1),
        nn.ReLU(),

        nn.Conv2d(512,512,kernel_size=3,padding=1),
        nn.ReLU(),

        nn.Conv2d(512,512,kernel_size=3,padding=1), # (16,16,512)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2), # (8,8,512)
    )



    VGG16_Output = nn.Sequential(
        nn.Linear(8, 128),
        nn.ReLU()
    )


    #x = conv1(input)
    #x = conv2(x)
    #x = conv3(x)
    #x = x.view(x.size()[0], -1)
    #x = fc1(x)
    #x = fc2(x)

    x = VGG16_Conv1(input)
    x = VGG16_Conv2(x)
    x = VGG16_Conv3(x)
    x = VGG16_Conv4(x)
    x = VGG16_Output(x)

    return x


def image_CNN(n,image_base_path):
    result = np.array([])  # 创建一个空的一维数组
    print("开始使用CNN处理图像")
    T = 1
    for i in os.listdir(image_base_path):
        image = get_picture(image_base_path + '/' + i, transform)
        image = image.unsqueeze(0)

        time_start = time.time()

        # 使用CNN对图像进行特征提取
        image = CNN_Model(image)
        # 展平提取后的图像
        image_arr_temp = image.flatten(start_dim=0)
        image_arr_temp = image_arr_temp.detach().numpy()

        result = np.concatenate((result, image_arr_temp))

        time_end = time.time()
        time_used = time_end - time_start
        print("第", T, "张图片的处理时间为", time_used)
        T += 1

    result = result.reshape((n, 524288))

    return result
















