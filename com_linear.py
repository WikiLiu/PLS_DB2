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
# 假设 X 为包含自变量的数据框，Y 为因变量的数据框
# 假设有 30 个正常样本和 1 个异常样本
X_normal = X.iloc[:-1, :]
Y_normal = y.iloc[:-1]
X_outlier = X.iloc[-1:, :]
Y_outlier = y.iloc[-1]

# 训练多元线性回归模型
reg = LinearRegression().fit(X_normal, Y_normal)
y_pred = reg.predict(X_outlier)
# 计算每个自变量对因变量的影响
effects = []
var_effects = {}
for i in range(X.shape[1]):
    X_train = X_normal.drop(columns=[i])
    reg_i = LinearRegression().fit(X_train, Y_normal)
    y_pred_i = reg_i.predict(X_outlier.drop(columns=[i]))
    effect = abs(y_pred.max() - y_pred_i.max()) / abs(Y_normal.mean() - y_pred_i.max())
    var_effects[f"Variable {i+1}"] = effect[0]
    effects.append(effect[0])

# 找到影响最大的自变量
max_effect = max(var_effects.values())
max_var = [var for var, effect in var_effects.items() if effect == max_effect]

# 输出结果
print(f"The outlier in the dependent variable is likely caused by {max_var[0]}.")


print(effects)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.8
ax.bar(range(len(effects)), effects, width=bar_width)

ax.set_xticks(range(len(effects)))
ax.set_xticklabels([str(i+1) for i in range(len(effects))], fontsize=10)

plt.show()

print(top30_features)











import matplotlib.pyplot as plt
import numpy as np

# 假设你已经有了数据和横坐标
x_labels = ['ENTRY_TENSION_6', 'CHEM_COEFF_7', 'ROLL_DIAM_7', 'KM_7', 'DELTA_WATER',
       'WATER_FLOW', 'CORR_ZEROPOINT_USE_6', 'MILLSTRETCH_ROLL_6',
       'FET_ACT_TEMP_7', 'ENTRY_THICK_7', 'DELTA_SPEED_7', 'DETAL_FORCE_CAL_6',
       'CORR_ZEROPOINT_USE_7', 'DELTA_REDU_7', 'TEMP_CORR_6', 'GAP_DELTA_6',
       'DELTA_MILL_7', 'DELTA_MILL_6', 'KM_6', 'TEMP_CORR_7',
       'CORR_FORCE_STAND_7', 'FORCE_ACT_7', 'LSAT_DELTA_TEMP',
       'DETAL_FORCE_POST_7', 'TEMP_DELTA_6', 'DELTA_CLASS', 'FM_TEMP_7',
       'DESC_SUM_7', 'TEMP_DELTA_7', 'DETAL_FORCE_POST_6']
# 绘制条形图
fig, ax = plt.subplots()
ax.barh(range(len(effects)), effects)
ax.set_yticks(range(len(effects)))
ax.set_yticklabels(x_labels)

# 找到前二高的条形并添加标注
sorted_indices = np.argsort(effects)[::-1] # 按照从大到小的顺序排序索引
for i in range(2):
    index = sorted_indices[i]
    height = effects[index]
    label = "宽度" if i == 0 else "厚度"
    ax.text(height+0.001, index, label, ha='left', va='center')

plt.show()

