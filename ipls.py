from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

df = pd.read_csv("output.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
data_small = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_small = data_small.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)
Y_small = data_small["DELTA_THICK_7"]
x_scaled = preprocessing.scale(X_small)

df = pd.read_csv("output_pre.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_big = data_big.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)
Y_big = data_big["DELTA_THICK_7"]
# 导入数据并进行均值方差归一化
X_scaled = preprocessing.scale(X_big)
# Y_scaled = preprocessing.scale(Y)
# 打乱顺序
X_shuffle, Y_shuffle = shuffle(X_scaled, Y_big, random_state=0)
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

# # 用大类数据集训练PLS模型
# pls.fit(data_big.iloc[:, :-1], data_big.iloc[:, -1])

# 获取模型的系数和截距
coefficients = pls.coef_
intercept = pls._y_mean - np.dot(pls._x_mean, coefficients)

# 对小类数据集进行预测
predictions = np.dot(x_scaled, coefficients) + intercept
pre2 = pls.predict(x_scaled)
# 计算预测误差
errors = data_small.iloc[:, -1] - predictions

# 使用小类数据集对模型参数进行拟合
pls_final = PLSRegression(n_components=2)
pls_final.fit(data_small.iloc[:, :-1], data_small.iloc[:, -1])
coefficients_final = pls_final.coef_
intercept_final = pls_final._y_mean - np.dot(pls_final._x_mean, coefficients_final) + np.mean(errors)
# 使用最终的模型参数对数据进行预测
predictions_final = np.dot(data_small.iloc[:, :-1], coefficients_final) + intercept_final
