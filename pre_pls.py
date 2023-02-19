import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.cross_decomposition import PLSRegression
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

# 定义偏最小二乘回归模型
pls = PLSRegression(scale=True)

# 交叉验证选择最佳的组分数量
n_components = np.arange(1, 21)
scores = []
for n in n_components:
    pls.set_params(n_components=n)
    score = model_selection.cross_val_score(pls, X_train, Y_train, cv=10, scoring='neg_mean_absolute_error')
    scores.append(-score.mean())
best_n = n_components[np.argmin(scores)]

# 使用最佳的组分数量训练模型
pls.set_params(n_components=best_n)
pls.fit(X_train, Y_train)

# 预测测试集结果
Y_pred = pls.predict(X_test)

# 计算预测精度
score = -mean_absolute_error(Y_test, Y_pred)
print('偏最小二乘模型预测精度（L1 loss）：', score,best_n)

import matplotlib.pyplot as plt
plt.scatter( Y_pred, Y_test)
plt.show()








from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression

# 定义参数范围
param_grid = {'n_components': [2, 3, 4, 5],
              'scale': [True, False],
              'max_iter': [100, 200, 300],
              'tol': [1e-3, 1e-4, 1e-5],
              'copy': [True, False]}

# 初始化模型和网格搜索
pls = PLSRegression()
grid_search = GridSearchCV(pls, param_grid, cv=5, scoring='neg_mean_squared_error')

# 在训练集上进行网格搜索
grid_search.fit(X_train, Y_train)

# 打印最佳超参数和最佳交叉验证得分
print('Best parameters:', grid_search.best_params_)
print('Best CV score:', -grid_search.best_score_)

# 使用最佳超参数重新训练模型并预测测试集
best_pls = grid_search.best_estimator_
Y_pred = best_pls.predict(X_test)

# 计算预测精度
score = -mean_absolute_error(Y_test, Y_pred)
print('L1 loss:', score)



