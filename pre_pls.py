import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("output_pre.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]

df = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)



X = df.drop(["DELTA_THICK_7","STRIP_NO_6"],axis=1)
Y = df["DELTA_THICK_7"]




# 导入数据并进行均值方差归一化
X_scaled = preprocessing.scale(X)
# Y_scaled = preprocessing.scale(Y)

# 打乱顺序
X_shuffle, Y_shuffle = shuffle(X_scaled, Y, random_state=0)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_shuffle, Y_shuffle, test_size=0.3, random_state=0)

# 训练偏最小二乘回归模型
pls = cross_decomposition.PLSRegression(n_components=25)
pls.fit(X_train, Y_train)

# 预测测试集结果
Y_pred = pls.predict(X_test)

# 计算预测精度
score = mean_absolute_error(Y_test, Y_pred)
print('偏最小二乘模型预测精度：', score)



import matplotlib.pyplot as plt
plt.scatter( Y_pred, Y_test)
plt.show()


