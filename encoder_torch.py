import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

strip_no = "220195003500"
df = pd.read_csv("output_pre220199004300.csv", index_col=0)
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
# 读取列名
with open('keep_feature.txt', 'r') as f:
    column_names = f.read().splitlines()
# data_big = data_big[column_names]
X = data_big.drop(["STRIP_NO_6"], axis=1)
names = list(X)
X_= preprocessing.scale(X)
# 加载数据集
train_data = X_

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=36, encoding_dim=20):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
# 训练集
train_dataset = MyDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型、损失函数和优化器
model = AutoEncoder(input_dim = len(names))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs = data.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

sample = torch.from_numpy(train_data).float()
X_pred = model(sample)



scored = np.mean((X_pred.detach().numpy()-train_data), axis = 1)
plt.figure()
sns.distplot(scored,
             bins = 10,
             kde= True,
            color = 'blue')
# plt.xlim([0.0,.5])
plt.show()
data = [['scored'] + [scored[i] for i in range(len(scored))]]
# 转置为按列的嵌套列表
data_T = list(map(list, zip(*data)))
import csv
# 写入CSV文件
with open('r_Loss_mae.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_T)



df = pd.read_csv("output"+strip_no+".csv", index_col=0)
id_to_match = strip_no
matched_rows = df.query(f'STRIP_NO_6 == ' + id_to_match)
df = pd.concat([df.drop(matched_rows.index), matched_rows])
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
# 读取列名
with open('keep_feature.txt', 'r') as f:
    column_names = f.read().splitlines()
# data_big = data_big[column_names]
X = data_big.drop(["STRIP_NO_6"], axis=1)
# 导入数据并进行均值方差归一化
x= preprocessing.scale(X)
# Y_scaled = preprocessing.scale(Y)
# 生成数据
X_normal = x[:-1,:]
X_abnormal = x[-1,:]
X_abnormal = X_abnormal[np.newaxis, :]
train_data = X_normal
# 训练集
train_dataset = MyDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 训练模型
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs = data.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

# 进行根因分析
model.eval()
# 检测异常
sample = torch.from_numpy(X_abnormal).float()
decoded_sample = model(sample)
reconstruction_errors = np.abs(decoded_sample.detach().numpy() - sample.detach().numpy() )[0]
# 计算重构误差
print(f"异常样本偏差度：{np.mean(reconstruction_errors)}")
# 找出最大偏差所在的特征
max_error_feature = np.argmax(reconstruction_errors)
print("The abnormality is likely caused by feature", names[max_error_feature])
import matplotlib.pyplot as plt
# 画直方图
# 创建子图
# 创建figure对象并设置画布大小
fig = plt.figure(figsize=(8, 10))
# 创建子图并绘制条形图
ax = fig.add_subplot(111)
# 设置x轴标签和位置
ax.set_xlabel('Features')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=90)
reconstruction_errors = reconstruction_errors.reshape(len(names))
# 绘制条形图
ax.bar(names, reconstruction_errors)
# 设置标签字体大小
plt.xticks(fontsize=8)
# 显示图形
plt.show()
# 对数据进行排序并获取索引
idx = np.argsort(reconstruction_errors)[::-1]
# 根据索引重新排列names
sorted_names = [names[i] for i in idx]
# 打印排序后的names
print(sorted_names)

