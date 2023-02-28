import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing

strip_no = "220199300200"

df = pd.read_csv("output_pre"+".csv", index_col=0)
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
# 读取列名
with open('keep_feature.txt', 'r') as f:
    column_names = f.read().splitlines()
data_big = data_big[column_names]
X = data_big.drop(["STRIP_NO_6"], axis=1)
names = list(X)
X= preprocessing.scale(X)
# 加载数据集
data = X

# 构建自编码器模型
input_dim = data.shape[1]
encoding_dim = 10

input_layer = keras.Input(shape=(input_dim,))
encoder_layer_1 = keras.layers.Dense(20, activation='relu')(input_layer)
encoder_layer_2 = keras.layers.Dense(encoding_dim, activation='relu')(encoder_layer_1)
decoder_layer_1 = keras.layers.Dense(20, activation='relu')(encoder_layer_2)
decoder_layer_2 = keras.layers.Dense(input_dim)(decoder_layer_1)

autoencoder = keras.Model(input_layer, decoder_layer_2)
# 训练自编码器模型
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(data, data, epochs=100, batch_size=32)

df = pd.read_csv("output"+strip_no+".csv", index_col=0)
id_to_match = strip_no
matched_rows = df.query(f'STRIP_NO_6 == ' + id_to_match)
# error = df.query(f'DELTA_THICK_7 > ' + "0.1" + " | DELTA_THICK_7 < -0.1")
# df = df.drop(error.index)
# df = pd.concat([df, matched_rows])
df = pd.concat([df.drop(matched_rows.index), matched_rows])
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
# 读取列名
with open('keep_feature.txt', 'r') as f:
    column_names = f.read().splitlines()
data_big = data_big[column_names]
X = data_big.drop(["STRIP_NO_6"], axis=1)
# 导入数据并进行均值方差归一化
x= preprocessing.scale(X)
# Y_scaled = preprocessing.scale(Y)
# 生成数据
X_normal = x[:-1,:]
X_abnormal = x[-1,:]
X_abnormal = X_abnormal[np.newaxis, :]

data = X_normal

autoencoder.fit(data, data, epochs=100, batch_size=32)

# 提取编码器模型
encoder = keras.Model(input_layer, encoder_layer_2)

# 检测异常
sample = X_abnormal
encoded_sample = encoder.predict(sample)
decoded_sample = autoencoder.predict(sample)
reconstruction_errors = np.abs(decoded_sample - sample)[0]
# 计算重构误差
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
reconstruction_errors = reconstruction_errors.reshape(data.shape[1])
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

