 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_decomposition, model_selection, preprocessing

df = pd.read_csv("output.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]

df = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_small = df.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)

# x_scaled = preprocessing.normalize(X_small)

from sklearn.preprocessing import MinMaxScaler,StandardScaler

# 创建MinMaxScaler对象
scaler = StandardScaler()

# 对数据进行0-1归一化
X_small = scaler.fit_transform(X_small)



names = df.columns.values

index = list(names).index("FET_ACT_TEMP_7")
FET_ACT_TEMP_7 = X_small[index]


y = df["DELTA_THICK_7"]


cor = df.corr()
cor_target = (cor["DELTA_THICK_7"])
cor_name = df.columns.tolist()
cor_data = cor_target.values
# 将一维数据转化为二维数据
cor_data = cor_data.reshape((1, -1))
# 绘制热力图
fig, ax = plt.subplots()
im = ax.imshow(cor_data, cmap='hot', interpolation='nearest')
# 设置横轴和纵轴的显示范围
ax.set_xlim(0, len(cor_data[0]))
ax.set_ylim(0, 1)
# 隐藏横轴和纵轴
ax.set_xticks([])
ax.set_yticks([])
# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
plt.show()

#Correlation with output variable

#Selecting highly correlated features
cor_target.sort_values(ascending=False)

pd.DataFrame(cor_target).columns.values.tolist()
relevant_features = cor_target.sort_values()
print(relevant_features)

keep_feature = []
with open("keep_feature.txt",'w') as f:
    for i, v in relevant_features.iteritems():
        if(abs(v)>0.1):
            f.write(i+"\n")
            keep_feature.append(i)

keep_feature = keep_feature[0:-1]
df = df[keep_feature]

from sklearn.linear_model import LinearRegression
import numpy as np

# 创建并拟合线性回归模型
model = LinearRegression(fit_intercept=True)
model.fit(X_small, y)

# 输出模型的参数（系数）和截距
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# 计算自变量的重要度评分
importance_scores = np.abs(model.coef_) / np.sum(np.abs(model.coef_))
cor_name[1:-1]

for i in range(44):
    print([str(cor_name[i]) + ":" + str(importance_scores[i])    ])
print('Importance scores:', importance_scores)


print("ok")