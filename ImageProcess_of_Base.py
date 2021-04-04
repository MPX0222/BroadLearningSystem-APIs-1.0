# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:26:20 2021

@author: MPX
"""
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import PIL.Image as Image
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

os.environ['KERAS_BACKEND']='tensorflow'


# RGB图像展平（n为数据总个数，path为图片文件夹地址）
def image_to_matrix_rgb(n,image_base_path):
    
        result = np.array([])  # 创建一个空的一维数组
        print("开始将图片转为数组")
        for i in os.listdir(image_base_path):
            image = Image.open(image_base_path + '\\' + i)
            r, g, b = image.split()  # rgb通道分离
            
            # reshape的参数 = 图片长度 × 图片宽度
            r_arr = np.array(r).reshape(16384)/255 # 归一化
            g_arr = np.array(g).reshape(16384)/255
            b_arr = np.array(b).reshape(16384)/255
            
            # 行拼接，最终结果：共n行，一行3072列
            image_arr = np.concatenate((r_arr, g_arr, b_arr))
            result = np.concatenate((result, image_arr))
            
        # 这里reshape的参数 = 三通道数组长度之和
        result = result.reshape((n, 49152))
        
        return result

# 标签读取（n为数据总个数，m为数据类别数，filedir为图片文件夹地址）
def label_to_matrix(n,m,filedir):
    
    labellist = os.listdir(filedir)
   
    mat = np.zeros(shape = (n,m))
    j = 0
    for i in labellist:
        index = i.find("_", 0)
        tem = int(i[index + 1])
        mat[j][tem] = 1
        j = j + 1
     
    return mat

# 灰度图像展平（n为数据总个数，path为图片文件夹地址）
def image_to_matrix_gray(n,image_base_path):

    rgb_tran_gray(image_base_path)

    result = np.array([])  # 创建一个空的一维数组
    print("开始将图片转为数组")
    for i in os.listdir(image_base_path):
        image = Image.open(image_base_path + '\\' + i)

        image_arr_temp = np.array(image).reshape(16384) / 255

        # 行拼接，最终结果：共n行，一行3072列
        result = np.concatenate((result, image_arr_temp))

    # 这里reshape的参数 = 三通道数组长度之和
    result = result.reshape((n, 16384))

    return result

# 将RGB图像转换为灰度图
def rgb_tran_gray(image_base_path):

    for i in os.listdir(image_base_path):
        img = Image.open(image_base_path + '/' + i)
        img_gray = img.convert('L')
        img_gray.save(image_base_path + '/' + i)

# 图像增强（只能独立调用）
def image_Enhancement(image_base_path):

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    for i in os.listdir(image_base_path):
        img = load_img(image_base_path + '/' + i)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=image_base_path, save_prefix=i, save_format='jpg'):
            i += 1
            if i > 4:
                break




