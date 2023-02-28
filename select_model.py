import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("output_pre220199004300.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_big = data_big.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)
Y_big = data_big["DELTA_THICK_7"]
# 导入数据并进行均值方差归一化
# X_scaled = preprocessing.scale(X_big)
# Y_scaled = preprocessing.scale(Y)
# 打乱顺序
X_shuffle, Y_shuffle = shuffle(X_big, Y_big, random_state=0)
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_shuffle, Y_shuffle, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 定义偏最小二乘回归模型
pls_base = PLSRegression(scale=True)
# pls_base = RandomForestRegressor(n_estimators=100, random_state=42)

# 交叉验证选择最佳的组分数量

n_components = np.arange(2,20)
scores = []
for n in n_components:
    pls_base.set_params(n_components=n)
    score = model_selection.cross_val_score(pls_base, X_train, Y_train, cv=10, scoring='r2')
    scores.append(score.mean())
best_n = n_components[np.argmax(scores)]
# 使用最佳的组分数量训练模型
pls_base.set_params(n_components=best_n)
pls_base.fit(X_train, Y_train)
print(f"best_n:{best_n}\n")
# 预测测试集
y_pred = pls_base.predict(X_test)
# 计算R2分数
r2 = r2_score(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
print(f"PLS回归模型的R2分数为：{r2:.4f}")
print(f"MAE: {mae:.2f}")


#
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.metrics import mean_squared_error
# # 构建模型
# model = Sequential()
# model.add(Dense(64, input_dim=43, activation='relu'))
# model.add(Dense(32, input_dim=64, activation='relu'))
# model.add(Dense(1))
#
# # 编译模型
# model.compile(loss='mean_squared_error', optimizer='adam')
# # 训练模型
# model.fit(X_train, Y_train, epochs=100)
#
# # 预测
# y_pred = model.predict(X_test)
#
# # 输出模型评估指标
# mae = mean_absolute_error(Y_test, y_pred)
# r2 = r2_score(Y_test, y_pred)
#
# # 输出评价指标
# print(f"Bayesian Ridge Regression:")
# print(f"MAE: {mae:.5f}")
# print(f"R2: {r2:.5f}")
#
# import matplotlib.pyplot as plt
#
# # 绘制散点图
# plt.scatter(Y_test, y_pred)
#
# # 添加标签和标题
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True vs. Predicted Values')
#
# # 显示图形
# plt.show()
#
# # 计算每个特征对预测结果的重要性
# sensitivity = []
# for i in range(X_big.shape[1]):
#     X_temp = X_test.copy()
#     np.random.shuffle(X_temp[:, i])
#     y_pred_temp = model.predict(X_temp)
#     sensitivity.append(np.mean(np.abs(y_pred_temp - y_pred)))
#
# # 特征重要性排序
# idx = np.argsort(sensitivity)[::-1]
# sorted_sensitivity = np.array(sensitivity)[idx]
#
# # 可视化特征重要性排序结果
# plt.bar(range(X_big.shape[1]), sorted_sensitivity)
# plt.xticks(range(X_big.shape[1]), idx)
# plt.xlabel('Feature Index')
# plt.ylabel('Feature Sensitivity')
# plt.show()
#
#
#
#
#












import torch
from torch.autograd import Variable

# 定义神经网络模型
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

# 定义模型参数
n_input = 43
n_hidden = 62
n_output = 1

# 生成一些随机的输入数据
x = torch.randn(100, n_input)

# 计算输出数据
net = Net(n_input, n_hidden, n_output)
y = net(x)

# 计算每个特征的梯度平方和
gradients = torch.autograd.grad(y.sum(), net.parameters(), create_graph=True)
gradients_squared = [g.pow(2) for g in gradients]
feature_importance = [g.mean() for g in gradients_squared]

# 输出特征重要性排名
sorted_importance, sorted_idx = torch.sort(torch.tensor(feature_importance), descending=True)
for i in range(len(sorted_idx)):
    print(f"Feature {sorted_idx[i]}: {sorted_importance[i]}")



# # 输出特征重要性评分
# scaler = StandardScaler()
# X_train_large_scaled = scaler.fit_transform(X_train_large)
# X_train_small_scaled = scaler.transform(X_train_small)
#
# f_statistic, p_values = f_regression(X_train_large_scaled, Y_train_large)
# feature_importance_base = f_statistic / np.sum(f_statistic)
#
# f_statistic, p_values = f_regression(X_train_small_scaled, Y_train_small)
# feature_importance_tuned = f_statistic / np.sum(f_statistic)
#
# print('特征重要性评分（基模型）：', feature_importance_base)
# print('特征重要性评分（微调后）：', feature_importance_tuned)


#
# import numpy as np
#
# # 生成一维数据
# x = np.random.rand(100)
# y = 2 * x + np.random.randn(100) * 0.1
#
# # 计算均值和标准差并标准化
# x_mean = np.mean(x)
# x_std = np.std(x)
# # x_std[0] = 1
# x_norm = (x - x_mean) / x_std
#
# y_mean = np.mean(y)
# y_std = np.std(y)
# # y_std[y_std == 0] = 1
# y_norm = (y - y_mean) / y_std
#
# # 计算 X 和 Y 的协方差矩阵和 X 的方差
# cov_xy = np.sum(x_norm * y_norm) / len(x_norm)
# var_x = np.sum(x_norm ** 2) / len(x_norm)
#
# # 计算 X 的主成分系数和 Y 的主成分系数，以及 X 和 Y 的载荷系数
# x_loadings = cov_xy / var_x
# y_loadings = 1
#
# # 建立 PLSR 模型并预测 Y 值
# y_pred = x_norm * x_loadings * y_std + y_mean
#
# import matplotlib.pyplot as plt
# import numpy as np
# # 创建一个新的图形
# plt.figure()
#
# # 绘制预测值和实际值的曲线
# plt.plot(y_pred, label='Predicted')
# plt.plot(y, label='True')
#
# # 添加图例、轴标签和标题
# plt.legend()
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.title('Predicted vs True')
#
# # 显示图形
# plt.show()