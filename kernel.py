import numpy as np
from sklearn import kernel_approximation, model_selection, preprocessing
from sklearn.cross_decomposition import PLSCanonical
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
import pandas as pd
from sklearn.kernel_approximation import Nystroem


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

# 使用RBF核将输入数据映射到高维空间
rbf_feature = kernel_approximation.RBFSampler(gamma=1, n_components=100)
X_train_kernel = rbf_feature.fit_transform(X_train)
X_test_kernel = rbf_feature.transform(X_test)


# # 使用Nystroem核将输入数据映射到高维空间
# nystroem_feature = Nystroem(gamma=1, n_components=100)
# X_train_kernel = nystroem_feature.fit_transform(X_train)
# X_test_kernel = nystroem_feature.transform(X_test)


# # 定义多项式核
# poly_kernel = np.polynomial.polynomial.Polynomial([0, 1])
# # 使用多项式核将输入数据映射到高维空间
# X_train_kernel = poly_kernel(X_train)
# X_test_kernel = poly_kernel(X_test)


# 训练核偏最小二乘回归模型
kpca = PLSCanonical(n_components=1)
kpca.fit(X_train_kernel, Y_train)

# 预测测试集结果
Y_pred = kpca.predict(X_test_kernel)

# 计算预测精度
score = mean_absolute_error(Y_test, Y_pred)
print('核偏最小二乘模型预测精度（L1 loss）：', score)
