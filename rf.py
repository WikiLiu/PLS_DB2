from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import cross_decomposition, model_selection, preprocessing
import pandas as pd
# 生成示例数据
df = pd.read_csv("output220206104400.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
data_small = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_small = data_small.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1).values

Y_small = data_small["DELTA_THICK_7"].values
x_scaled = preprocessing.scale(X_small)

names = list(data_small.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1))


df = pd.read_csv("output_pre.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_big = data_big.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1).values
Y_big = data_big["DELTA_THICK_7"].values

# 定义随机森林回归模型
rf = RandomForestRegressor(n_estimators=100)

# 使用大类数据集训练随机森林模型
rf.fit(X_big, Y_big)

# 对小类数据集进行预测并获取袋外预测值
oob_pred = rf.predict(X_small)

# 打乱每一维的数据，计算变量重要性
feat_imp = np.zeros((43,))
for i in range(43):
    sum = 0
    for j in range(5):
        temp_data = X_small.copy()
        np.random.shuffle(temp_data[i, :])
        oob_pred_shuffled = rf.predict(temp_data)
        sum += np.mean((oob_pred - oob_pred_shuffled)**2)
    feat_imp[i] = sum/5
# 输出变量重要性
print(feat_imp)


import matplotlib.pyplot as plt

# 将名称从1开始排列
labels = names

# 绘制旋转90度的条形图
fig, ax = plt.subplots(figsize=(8,12))
ax.barh(labels, feat_imp, height=0.8)

# 添加x轴标签
ax.set_xlabel('Feature Importance')

# 旋转刻度标签
ax.set_yticklabels(labels)

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()