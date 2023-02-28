import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("output220195003500.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]

df = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)

id_to_match = "220195003500"
matched_rows = df.query(f'STRIP_NO_6 == ' + id_to_match)
# error = df.query(f'DELTA_THICK_7 > ' + "0.1" + " | DELTA_THICK_7 < -0.1")
# df = df.drop(error.index)
df = pd.concat([df.drop(matched_rows.index), matched_rows])

# # 获取最后一行数据
# last_row = df.iloc[[-1]]
#
# # 打乱数据，不包括最后一行
# df = df.iloc[:-1].sample(frac=1, random_state=123).reset_index(drop=True)
#
# # 将最后一行数据添加到最后
# df = pd.concat([df, last_row], axis=0).reset_index(drop=True)

X = df.drop(["STRIP_NO_6"],axis=1)
# 计算所有特征和第一列的皮尔逊相关系数
# corr_matrix = X.corr(method='pearson')

# 排序并选择相关性前30的特征
# top30_features = corr_matrix.iloc[1:, 0].abs().sort_values(ascending=False).index[:31]
top30_features = list(X)
# 删除不需要的特征
# X = X[top30_features]





Y = df["DELTA_THICK_7"]
Y = np.array(Y)



# 导入数据并进行均值方差归一化
X_scaled = preprocessing.scale(X)
Y_scaled = preprocessing.scale(Y)

# 打乱顺序
# X_shuffle, Y_shuffle = shuffle(X_scaled, Y, random_state=0)

# 划分训练集和测试集
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_scaled, Y, test_size=0.3)
from factor_analyzer import FactorAnalyzer
# 创建因子分析对象并指定要使用的因子数量
fa = FactorAnalyzer(n_factors=5, method="ml", rotation="varimax")

# 对自变量进行因子分析
fa.fit(X)
# 获取因子载荷矩阵
loadings = fa.loadings_

# 打印每个自变量与每个因子的相关性
for i in range(X.shape[1]):
    print(f"Variable {i+1}: {loadings[i]}")

# 获取因变量与每个因子的相关性
corr = np.corrcoef(X, rowvar=False)[-1, :-1]

for i in range(X.shape[1]-1):
    # 打印每个自变量与因变量的相关性for i in range(X.shape[1]):
    if abs(corr[i]) > 0.5:
        print(f"Variable {top30_features[i]} is highly correlated with the dependent variable.")

# 比较最后一个样本的自变量值与其他样本的自变量值
for i in range(X.shape[1]-1):
    if abs(corr[i]) > 0.5:
        var_name = f"Variable {top30_features[i]}"
        var_val = X.iloc[-1, i]
        var_mean = X.iloc[:, i].mean()
        if abs(var_val - var_mean) > 2 * X.iloc[:, i].std():
            print(var_name+"  "+str(var_val - var_mean))
            print(f"The outlier in the dependent variable is likely caused by {var_name}.")

# # 使用回归模型分析每个与因变量高度相关的自变量与因变量之间的关系
# from sklearn.linear_model import LinearRegression
#
# for i in range(X.shape[1]):
#     if abs(corr[i]) > 0.5:
#         var_name = f"Variable {i+1}"
#         X_train = X.iloc[:-1, i].values.reshape(-1, 1)
#         y_train = y.iloc[:-1].values.reshape(-1, 1)
#         X_test = X.iloc[[-1], i].values.reshape(-1, 1)
#         y_test = y.iloc[[-1]].values.reshape(-1, 1)
#         reg = LinearRegression().fit(X_train, y_train)
#         y_pred = reg.predict(X_test)
#         if abs(y_pred - y_test) > 2 * y.std():
#             print(f"The outlier in the dependent variable is likely caused by {var_name}.")


print(corr)
import matplotlib.pyplot as plt
plt.plot(corr)
plt.show()

# # 计算SPE和T2统计量
# X_scores, Y_scores = pls.transform(X, Y)
# SPE = np.sum((X - np.dot(X_scores, pls.x_weights_.T)) ** 2, axis=1)
# T2 = np.sum((X_scores / np.sqrt(np.sum(X_scores ** 2, axis=0))) ** 2, axis=1)
#
# # 打印SPE和T2统计量
# print("SPE统计量：", SPE)
# print("T2统计量：", T2)







# 初始化模型和网格搜索




