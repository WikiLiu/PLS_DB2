import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler

df = pd.read_csv("output220206104400.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]

df = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_small = df.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)
scaler = StandardScaler()
# 对数据进行0-1归一化
X_small = scaler.fit_transform(X_small)
names = list(df)
y = df["DELTA_THICK_7"].values
# 创建并拟合线性回归模型
model = PLSRegression(n_components=3)
model.fit(X_small, y)
spe = np.square(y - model.predict(X_small)).sum(axis=1)
# 绘制带红点的折线图
fig, ax = plt.subplots()
ax.plot(spe, 'b-')
ax.scatter(range(len(spe)), spe, c='r')

# 获取回归系数矩阵
coef = model.coef_

# 计算某一个样本在所有维度上对特征的贡献值
sample = X_small[-1,:].reshape(-1,1)
contributions = np.multiply(sample.reshape(43), coef.reshape(43))

# 对数据进行排序并获取索引
idx = np.argsort(contributions)[::-1]
# 根据索引重新排列names
sorted_names = [names[i] for i in idx]
# 打印排序后的names
print(sorted_names)
print("ok")