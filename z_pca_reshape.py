from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing

df = pd.read_csv("output220206104400.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
id_to_match = "220206104400"
matched_rows = df.query(f'STRIP_NO_6 == ' + id_to_match)
# error = df.query(f'DELTA_THICK_7 > ' + "0.1" + " | DELTA_THICK_7 < -0.1")
# df = df.drop(error.index)
# df = pd.concat([df, matched_rows])
df = pd.concat([df.drop(matched_rows.index), matched_rows])
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X = data_big.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)
names = list(X)
y = data_big["DELTA_THICK_7"]
# 导入数据并进行均值方差归一化
x= preprocessing.scale(X)
# Y_scaled = preprocessing.scale(Y)

# 生成数据
X_normal = x[:-1,:]
X_abnormal = x[-1,:]
# 在最前面增加一个维度
X_abnormal = X_abnormal[np.newaxis, :]
# 使用PCA降维
pca = PCA(n_components=13)
pca.fit(X_normal)
X_normal_transformed = pca.transform(X_normal)
X_abnormal_transformed = pca.transform(X_abnormal)

# 获取每个主成分的方差贡献率
variance_ratio = pca.explained_variance_ratio_

# 对方差贡献率从大到小排序，得到主成分的重要性排序
importance_order = np.argsort(variance_ratio)[::-1]

# 输出每个主成分的方差贡献率和重要性排序
for i, ratio in enumerate(variance_ratio):
    print(f"Principal component {i+1}: explained variance ratio={ratio:.3f}, importance order={np.where(importance_order == i)[0][0]+1}")
print(sum(variance_ratio))

# 逆向重构异常样本
X_abnormal_reconstructed = pca.inverse_transform(X_abnormal_transformed)
# 计算重构误差
reconstruction_errors =abs(X_abnormal - X_abnormal_reconstructed)
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
reconstruction_errors = reconstruction_errors.reshape(43)
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
#
#
#
#
#
# # 计算所有样本的SPE和T^2统计值
# spe = np.sum((X - pca.inverse_transform(pca.transform(X)))**2, axis=1).values
# t2 = np.sum((pca.transform(X) / np.sqrt(pca.explained_variance_))**2, axis=1)
#
# # 画SPE折线图
# plt.plot(spe, '-o')
# plt.xlabel('Sample index')
# plt.ylabel('SPE')
# plt.show()
#
# # 画T^2折线图
# plt.plot(t2, '-o')
# plt.xlabel('Sample index')
# plt.ylabel('T^2')
# plt.show()








