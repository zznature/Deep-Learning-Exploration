import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MyModel
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import AirPlaneDataset
import numpy as np
from utils import seed_torch

BATCH_SIZE = 32
EPOCHES = 20


def run(model, loss_func, optimizer, train_loader, test_loader):
    for epoch in range(EPOCHES):
        print(f'Epoch: {epoch}------------------------------------')
        for i, data in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()

            # 若使用gpu训练需要将tensor拷贝到gpu上
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = loss_func(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 每1000个batch打印loss等信息
            if i % 1000 == 0:
                print("epoch: %d, step: %d, Loss: %.3f" % (epoch, i, loss.item()))
    # 保存模型
    torch.save(net.state_dict(), './my_model_final.pt')
    return


if __name__ == '__main__':
    # 计算设备
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # 固定随机数种子
    seed_torch(31415926)
    ##############################################################################################
    # 数据集相关
    ##############################################################################################
    # 数据集预处理并加载
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    # 获取数据图片的所有路径
    all_imgs_path = glob.glob(r'D:\Airplane_Data_Classification\data\*\*.jpg')
    # 获取数据图片的所有标签
    species = ['up', 'down', 'left', 'right']
    species2 = np.eye(4)
    all_labels = []
    for img in all_imgs_path:
        for i, c in enumerate(species):
            if c in img:
                all_labels.append(species2[i, :])
    # 划分测试集和训练集
    index = np.random.permutation(len(all_imgs_path))
    all_imgs_path = np.array(all_imgs_path)[index]
    all_labels = np.array(all_labels)[index]
    # 前80%作为训练集
    s = int(len(all_imgs_path) * 0.8)
    train_imgs = all_imgs_path[:s]
    train_labels = all_labels[:s]
    test_imgs = all_imgs_path[s:]
    test_labels = all_labels[s:]
    # 构建Dataloader
    train_ds = AirPlaneDataset(train_imgs, train_labels, transform)  # TrainSet TensorData
    test_ds = AirPlaneDataset(test_imgs, test_labels, transform)  # TestSet TensorData
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # TrainSet Labels
    test_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # TestSet Labels
    ##############################################################################################

    # 定义模型
    net = MyModel().to(device)
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # 定义优化器
    optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 开始训练
    run(net, criterion, optim, train_dl, test_dl)

