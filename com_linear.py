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

from sklearn.linear_model import LinearRegression

# 训练回归模型
X = pd.DataFrame(X_scaled)
y = pd.DataFrame(Y_scaled)
reg = LinearRegression().fit(X.iloc[:-1,:], y.iloc[:-1])

# 找到异常值的索引
outlier_index = X.index[-1]

# 计算每个自变量对因变量的影响
var_effects = {}
for i in range(X.shape[1]):
    X_train = X.iloc[:-1,:]
    y_train = y.iloc[:-1]
    reg_i = LinearRegression().fit(X_train.drop(columns=[i]), y_train)
    y_pred_i = reg_i.predict(X_train.drop(columns=[i]))
    y_pred = reg.predict(X)
    effect = abs(y_pred - y_pred_i) / abs(y.mean() - y_pred_i)
    var_effects[f"Variable {i+1}"] = effect[0]

# 找到影响最大的自变量
max_effect = max(var_effects.values())
max_var = [var for var, effect in var_effects.items() if effect == max_effect]

# 输出结果
print(f"The outlier in the dependent variable is likely caused by {max_var[0]}.")



print(var_effects)
import matplotlib.pyplot as plt
plt.plot(var_effects)
plt.show()
