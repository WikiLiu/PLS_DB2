from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing

df = pd.read_csv("output220195003500.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
id_to_match = "220195003500"
matched_rows = df.query(f'STRIP_NO_6 == ' + id_to_match)
# error = df.query(f'DELTA_THICK_7 > ' + "0.1" + " | DELTA_THICK_7 < -0.1")
# df = df.drop(error.index)
df = pd.concat([df.drop(matched_rows.index), matched_rows])
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
# 读取列名
with open('keep_feature.txt', 'r') as f:
    column_names = f.read().splitlines()
# data_big = data_big[column_names]

X = data_big.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)





names = list(X)
y = data_big["DELTA_THICK_7"]
# 导入数据并进行均值方差归一化
x= preprocessing.scale(X)
# Y_scaled = preprocessing.scale(Y)

# 生成数据
normal_samples = x[:,:]
abnormal_sample = x[-1,:]
abnormal_sample = abnormal_sample[np.newaxis, :]
# 拟合 PCA 模型
pca = PCA(n_components=15)
pca.fit(normal_samples)

# SPE 统计值计算函数
def calculate_spe(sample):
    x_transformed = pca.transform(sample)
    x_reconstructed = pca.inverse_transform(x_transformed)
    spe = np.sum(np.square(sample - x_reconstructed), axis=1)
    return spe


# 计算异常样本 SPE 统计值
s0 = calculate_spe(abnormal_sample)

# 计算每一维的影响系数
influence_coeffs = []
for i in range(len(names)):
    # 将第 i 维变成 0
    modified_sample = np.copy(abnormal_sample)
    modified_sample[0][i] = 0

    # 计算 SPE 统计值
    s1 = calculate_spe(modified_sample)

    # 计算影响系数
    influence_coeff = s1 - s0
    influence_coeffs.append(influence_coeff[0])

# 找到影响最大的维度
max_influence_dim = np.argmax(np.abs(influence_coeffs))
root_cause = max_influence_dim + 1  # 由于 Python 下标从 0 开始，因此加 1 得到正确的维度

print(f"The root cause of the abnormality is dimension {root_cause}.")
# 将两个列表按列组合为一个二维列表
data = [['name', 'corr']] + [[names[i], influence_coeffs[i]] for i in range(len(influence_coeffs))]
import csv
# 写入CSV文件
with open('r_pca.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

import matplotlib.pyplot as plt
# 画直方图
# 创建子图
# 创建figure对象并设置画布大小
fig = plt.figure(figsize=(8, 10))
reconstruction_errors = influence_coeffs
# 创建子图并绘制条形图
ax = fig.add_subplot(111)
# 设置x轴标签和位置
ax.set_xlabel('Features')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=90)

# 绘制条形图
ax.bar(names, reconstruction_errors)

# 设置标签字体大小
plt.xticks(fontsize=8)
plt.title(id_to_match)
# 显示图形
plt.show()
# 对数据进行排序并获取索引
idx = np.argsort(reconstruction_errors)[::-1]

# 根据索引重新排列names
sorted_names = [names[i] for i in idx]

# 打印排序后的names
print(sorted_names)





