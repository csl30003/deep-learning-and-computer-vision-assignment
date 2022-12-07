import torch
import torch.nn as nn
import numpy as np
import os
import random

from load_img import read_train, read_test
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import Module, BCELoss


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding='same')
        self.max_pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(15 * 15 * 48, 720)
        self.fc2 = nn.Linear(720, 48)
        self.fc3 = nn.Linear(48, 1)
        self.last = nn.Sigmoid()

    # 前项传播
    def forward(self, x):
        # 240*240
        x = self.conv1(x)  # 240*240*6
        x = nn.ReLU(inplace=False)(x)
        x = self.max_pool3(x)  # 120*120*6

        x = self.conv2(x)  # 120*120*12
        x = nn.ReLU(inplace=False)(x)
        x = self.max_pool3(x)  # 60*60*12

        x = self.conv3(x)  # 60*60*24
        x = nn.ReLU(inplace=False)(x)
        x = self.max_pool3(x)  # 30*30*24

        x = self.conv4(x)  # 30*30*48
        x = nn.ReLU(inplace=False)(x)
        x = self.max_pool3(x)  # 15*15*48

        x = x.view(x.size(0), -1)

        x = self.fc1(x)  # 15*15*48>720
        x = nn.ReLU(inplace=False)(x)

        x = self.fc2(x)  # 720>48
        x = nn.ReLU(inplace=False)(x)

        x = self.fc3(x)  # 48>1
        x = self.last(x)

        return x


def df_model():
    # 定义模型
    model = Net()
    # 定义优化器
    optimizer = Adam(model.parameters(), lr=0.01)
    # 定义loss函数
    criterion = BCELoss()
    # criterion = BCEWithLogitsLoss()
    # 检查GPU是否可用
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    return model, optimizer, criterion


def train(epoch):
    # 训练
    model.train()
    tr_loss = 0
    # 获取训练集
    x_train, y_train = Variable(train_x), Variable(train_y)
    # 获取验证集
    x_val, y_val = Variable(val_x), Variable(val_y)
    # 转换为GPU格式
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # 清除梯度
    optimizer.zero_grad()

    # 预测训练与验证集
    output_train = model(x_train)
    output_val = model(x_val)

    output_train = output_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    output_val = output_val.to(torch.float32)
    y_val = y_val.to(torch.float32)
    # 计算训练集与验证集损失
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # 更新权重
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

    # 输出验证集loss
    print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)


def seed_torch(seed=30):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 设置随机种子
    seed_torch()

    # 优化运行效率
    torch.backends.cudnn.benchmark = True

    # 读数据集 注意没有全读 因为内存会炸
    img_path = '.\chest_xray'
    train_x, train_y = read_train(img_path)
    val_x, val_y = read_test(img_path)

    # 获取模型 优化器 损失函数
    model, optimizer, criterion = df_model()

    # 定义训练轮数
    n_epochs = 40

    # 空列表存储训练集损失
    train_losses = []
    # 空列表存储验证集损失
    val_losses = []

    # 训练模型
    for epoch in range(n_epochs):
        train(epoch)

    # 输出预测测试集的结构
    output = model(val_x)
    print("\n预测结果大于0.5为患有新冠肺炎 小于等于0.5为健康")
    print("预测结果为 ", end="")
    print(output)

    output = output.detach().numpy().copy()
    output = np.round(output).astype('int')

    print("验证集精度为 ", end="")
    print(accuracy_score(val_y, output))

    # with torch.no_grad():
    #     output = model(train_x.cuda())
    #
    # softmax = torch.exp(output).cpu()
    # prob = list(softmax.numpy())
    # predictions = np.argmax(prob, axis=1)
    #
    # # 训练集精度
    # print("训练集精度")
    # print(accuracy_score(train_y, predictions))
    #
    # # 验证集预测
    # with torch.no_grad():
    #     output = model(val_x.cuda())
    #
    # softmax = torch.exp(output).cpu()
    # prob = list(softmax.numpy())
    # predictions = np.argmax(prob, axis=1)
    #
    # # 验证集精度
    # print("验证集精度")
    # print(accuracy_score(val_y, predictions))
