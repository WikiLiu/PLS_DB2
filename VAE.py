import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing
import numpy as np
strip_no = "220199300200"
df = pd.read_csv("output_pre"+strip_no+".csv", index_col=0)
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
# 构建VAE模型
latent_dim = 2
# 定义编码器
inputs = keras.Input(shape=(input_dim,))
x = keras.layers.Dense(16, activation="relu")(inputs)
x = keras.layers.Dense(8, activation="relu")(x)
z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
# 定义采样函数
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
# 定义解码器
latent_inputs = keras.Input(shape=(latent_dim,))
x = keras.layers.Dense(8, activation="relu")(latent_inputs)
x = keras.layers.Dense(16, activation="relu")(x)
outputs = keras.layers.Dense(input_dim, activation="sigmoid")(x)

# 定义VAE模型
encoder = keras.Model(inputs, [z_mean, z_log_var, tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,))])
decoder = keras.Model(latent_inputs, outputs)
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs)

# 定义损失函数
reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= input_dim
kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# 编译模型
vae.compile(optimizer='adam')
# 训练模型
vae.fit(data, data, epochs=100, batch_size=32)




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

vae.fit(data, data, epochs=100, batch_size=32)


# 检测异常
sample = X_abnormal
decoded_sample = vae.predict(sample)
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
reconstruction_errors = reconstruction_errors.reshape(44)
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
