import numpy as np                # 导入NumPy库并将其别名为np
import torch                       # 导入PyTorch库
from torch.utils import data       # 从PyTorch的工具中导入数据工具包
from d2l import torch as d2l        # 导入d2l（动手学深度学习）中的PyTorch实用函数库并将其别名为d2l

true_w = torch.tensor([2, -3.4])   # 定义真实权重
true_b = 4.2                       # 定义真实偏置
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 使用d2l.synthetic_data函数生成特征和标签

def load_array(data_arrays, batch_size, is_train=True):  # 定义加载数据的函数
    dataset = data.TensorDataset(*data_arrays)           # 将特征和标签作为参数构造数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回数据加载器

batch_size = 10                    # 定义每个小批次的样本数
data_iter = load_array((features, labels), batch_size)  # 用数据加载器载入特征和标签

next(iter(data_iter))              # 查看第一批次的数据

from torch import nn               # 从PyTorch的神经网络模块中导入神经网络类
net = nn.Sequential(nn.Linear(2, 1))  # 构造一个包含单个线性层的网络模型，输入维度为2，输出维度为1
net[0].weight.data.normal_(0, 0.01)   # 使用正态分布随机初始化网络模型的权重参数
net[0].bias.data.fill_(0)            # 使用常数0初始化网络模型的偏置参数

loss = nn.MSELoss()                  # 定义均方误差损失函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 使用随机梯度下降优化器训练网络模型，学习率为0.03

num_epochs = 3                       # 定义训练轮数
for epoch in range(num_epochs):      # 对于每个训练轮数
    for X, y in data_iter:           # 对于每个小批次的样本
        l = loss(net(X) ,y)          # 计算网络模型的预测值与实际标签的损失
        trainer.zero_grad()          # 梯度清零
        l.backward()                 # 反向传播计算梯度
        trainer.step()               # 优化模型参数
    l = loss(net(features), labels)  # 计算完整数据集上的损失
    print(f'epoch {epoch + 1}, loss {l:f}')  # 输出损失

w = net[0].weight.data              # 得到网络模型的权重参数
print('w的估计误差：', true_w - w.reshape(true_w.shape))  # 输出权重参数的估计误

b = net[0].bias.data                      # 获取网络偏置
print('b的估计误差：', true_b - b)         # 计算偏置的估计误差并输出
