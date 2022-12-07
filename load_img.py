import os
import numpy as np
import cv2 as cv
import torch

from skimage.io import imread

train_img = []
train_label = []
test_img = []
test_label = []


def read_train(path):
    train_normal_path = path + '/train/NORMAL'
    train_pneumonia_path = path + '/train/PNEUMONIA'

    width = 240  # 2600
    height = 240  # 2300
    size = 300
    # 读训练集正常的图片
    files = os.listdir(train_normal_path)
    for file in files:
        image_path = train_normal_path + '/' + file
        # 读取图片
        img = imread(image_path, as_gray=True)
        img = cv.resize(img, (width, height))
        # print(type(img))
        # 归一化像素值
        img = img / 255.0
        # 转换为浮点数
        img = img.astype('float32')
        # 添加到列表
        train_img.append(img)
        train_label.append(0)
        if len(train_label) == 400:
            break

    # 读训练集不正常的图片
    files = os.listdir(train_pneumonia_path)
    for file in files:
        image_path = train_pneumonia_path + '/' + file
        # 读取图片
        img = imread(image_path, as_gray=True)
        img = cv.resize(img, (width, height))
        # 归一化像素值
        img = img / 255.0
        # 转换为浮点数
        img = img.astype('float32')
        # 添加到列表
        train_img.append(img)
        train_label.append(1)
        if len(train_label) == 600:
            break

    train_x = np.array(train_img)
    train_y = np.array(train_label).reshape(-1, 1)

    # 展示
    # i = 0
    # plt.figure(figsize=(10, 10))
    # plt.subplot(221), plt.imshow(train_x[i], cmap='gray')
    # plt.subplot(222), plt.imshow(train_x[i + 100], cmap='gray')
    # plt.subplot(223), plt.imshow(train_x[i + 400], cmap='gray')
    # plt.subplot(224), plt.imshow(train_x[i + 500], cmap='gray')
    # plt.show()

    per = np.random.permutation(train_x.shape[0])  # 打乱后的行号
    train_x = train_x[per, :, :]  # 获取打乱后的训练数据
    train_y = train_y[per, :]
    # 转换为torch张量
    train_x = train_x.reshape(2 * size, 1, width, height)
    train_x = torch.from_numpy(train_x)

    # 转换为torch张量
    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y)

    # print(train_x.shape)
    # print(train_y.shape)

    print("训练图片读取成功")

    return train_x, train_y


def read_test(path):
    test_normal_path = path + '/test/NORMAL'
    test_pneumonia_path = path + '/test/PNEUMONIA'

    width = 240  # 2600
    height = 240  # 2300
    size = 30

    # # 读测试集正常的图片
    files = os.listdir(test_normal_path)
    for file in files:
        image_path = test_normal_path + '/' + file
        # 读取图片
        img = imread(image_path, as_gray=True)
        img = cv.resize(img, (width, height))
        # print(type(img))
        # 归一化像素值
        img = img / 255.0
        # 转换为浮点数
        img = img.astype('float32')
        # 添加到列表
        test_img.append(img)
        test_label.append(0)
        if len(test_label) == size:
            break

    # # 读测试集不正常的图片
    files = os.listdir(test_pneumonia_path)
    for file in files:
        image_path = test_pneumonia_path + '/' + file
        # 读取图片
        img = imread(image_path, as_gray=True)
        img = cv.resize(img, (width, height))
        # 归一化像素值
        img = img / 255.0
        # 转换为浮点数
        img = img.astype('float32')
        # 添加到列表
        test_img.append(img)
        test_label.append(1)
        if len(test_label) == size * 2:
            break

    test_x = np.array(test_img)
    test_y = np.array(test_label).reshape(-1, 1)

    per = np.random.permutation(test_x.shape[0])  # 打乱后的行号
    test_x = test_x[per, :, :]  # 获取打乱后的训练数据
    test_y = test_y[per, :]

    # 转换为torch张量
    test_x = test_x.reshape(60, 1, width, height)
    test_x = torch.from_numpy(test_x)

    # 转换为torch张量
    test_y = test_y.astype(int)
    test_y = torch.from_numpy(test_y)

    print("测试图片读取成功")

    return test_x, test_y
