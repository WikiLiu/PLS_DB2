from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

df = pd.read_csv("output220206104400.csv", index_col=0)
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
pls_base = PLSRegression(scale=True)
# 交叉验证选择最佳的组分数量
n_components = np.arange(1, 21)
scores = []
for n in n_components:
    pls_base.set_params(n_components=n)
    score = model_selection.cross_val_score(pls_base, X_train, Y_train, cv=10, scoring='neg_mean_absolute_error')
    scores.append(-score.mean())
best_n = n_components[np.argmin(scores)]
# 使用最佳的组分数量训练模型
pls_base.set_params(n_components=best_n)
pls_base.fit(X_train, Y_train)


# 使用小类数据集微调基模型
pls_tuned = PLSRegression(n_components=3, scale=False, max_iter=500)
pls_tuned.set_params(coef_ = pls_base.coef_)
pls_tuned.x_rotations_ = pls_base.x_rotations_
pls_tuned.y_rotations_ = pls_base.y_rotations_
pls_tuned.x_scores_ = pls_base.transform(x_scaled)
pls_tuned.y_scores_ = pls_base.predict(x_scaled)
pls_tuned._center_scale_xy(x_scaled, Y_small, False)
pls_tuned._pls1_coef()

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





from sklearn.cross_decomposition import PLSRegression
import numpy as np

# 生成一个模拟数据集
X = np.random.normal(size=(100, 5))
y = np.random.normal(size=(100, 1))

# 训练一个偏最小二乘模型
pls = PLSRegression(n_components=2)
pls.fit(X, y)

# 生成新的数据集
X_new = np.random.normal(size=(50, 5))
y_new = np.random.normal(size=(50, 1))

# 使用新数据重新适应模型
pls.fit(X_new, y_new)

# 检查模型性能
score = pls.score(X_new, y_new)
print('R^2 score:', score)

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