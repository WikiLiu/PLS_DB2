import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("output.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]

df = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)

id_to_match = "220206104400"
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

X = df.drop(["DELTA_THICK_7","STRIP_NO_6"],axis=1)
# 计算所有特征和第一列的皮尔逊相关系数
corr_matrix = X.corr(method='pearson')

# 排序并选择相关性前30的特征
top30_features = corr_matrix.iloc[1:, 0].abs().sort_values(ascending=False).index[:30]

# 删除不需要的特征
X = X[top30_features]

# 输出处理后的数据
print(df.head())




Y = df["DELTA_THICK_7"]
Y = np.array(Y)



# 导入数据并进行均值方差归一化
X_scaled = preprocessing.scale(X)
Y_scaled = preprocessing.scale(Y)

# 打乱顺序
# X_shuffle, Y_shuffle = shuffle(X_scaled, Y, random_state=0)

# 划分训练集和测试集
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_scaled, Y, test_size=0.3)

# 定义偏最小二乘回归模型
pls = PLSRegression(scale=True)

# 交叉验证选择最佳的组分数量
n_ = np.arange(3, 4)
opt_n_components = 0
scores = []
spe = []
for n in n_:
    pls = PLSRegression(n_components=n)
    pls.fit(X_scaled, Y_scaled )
    Y_pre = pls.predict(X_scaled).squeeze()
    spe = abs(Y_pre - Y_scaled )**2
    # if spe[-1] == max(spe):
    #     opt_n_components = n
    #     break
    # else:
    #     continue

# 找到最大值的索引
max_index = np.argmax(spe)
# 将最大值从数组中取出
max_value = spe[max_index]
# 将最大值从原数组中删除
arr = np.concatenate([spe[:max_index], spe[max_index+1:]])
# 将最大值放到数组最后
arr = np.append(arr, max_value)
print(arr)
import matplotlib.pyplot as plt
plt.plot(arr)
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




