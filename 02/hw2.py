import numpy as np                       # 导入numpy库
import torch                              # 导入PyTorch库
from torch.utils import data             # 从torch.utils中导入data
from d2l import torch as d2l              # 从d2l库中导入torch模块并重命名为d2l
true_w = torch.tensor([2, -3.4])          # 定义一个张量true_w -> 参数w的真实值
true_b = 4.2                             # 定义一个偏置项 -> 偏置值bias的真实值
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 生成合成数据集
def load_array(data_arrays, batch_size, is_train=True):  # 定义一个函数load_array，构造一个PyTorch数据迭代器
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)   # 将数据数组组成数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 构造数据迭代器并返回
batch_size = 10                            # 定义批量大小
data_iter = load_array((features, labels), batch_size)   # 调用load_array函数构造数据迭代器
next(iter(data_iter))                      # 调用迭代器的iter()函数，并调用next()函数获取迭代器的下一个元素
from torch import nn                       # 从torch中导入nn模块
net = nn.Sequential(nn.Linear(2, 1))      # 构建一个全连接神经网络
net[0].weight.data.normal_(0, 0.01)        # 初始化网络权重
net[0].bias.data.fill_(0)                 # 初始化网络偏置
loss = nn.MSELoss()                        # 定义损失函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)   # 定义优化器
num_epochs = 3                             # 定义迭代次数
for epoch in range(num_epochs):            # 迭代训练
    for X, y in data_iter:
        l = loss(net(X) ,y)               # 计算损失
        trainer.zero_grad()               # 梯度清零
        l.backward()                      # 反向传播
        trainer.step()                    # 更新参数
    l = loss(net(features), labels)       # 计算训练误差
    print(f'epoch {epoch + 1}, loss {l:f}')   # 打印迭代次数和训练误差
w = net[0].weight.data                    # 获取网络权重
print('w的估计误差：', true_w - w.reshape(true_w.shape))   # 计算权重的估计误差并输出
b = net[0].bias.data                      # 获取网络偏置
print('b的估计误差：', true_b - b)         # 计算偏置的估计误差并输出
