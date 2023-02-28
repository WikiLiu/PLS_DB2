from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_decomposition, model_selection, preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

df = pd.read_csv("output_pre.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X_big = data_big.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)
Y_big = data_big["DELTA_THICK_7"]
# 导入数据并进行均值方差归一化
X_scaled = preprocessing.scale(X_big)
# Y_scaled = preprocessing.scale(Y)
# 打乱顺序
X_shuffle, Y_shuffle = shuffle(X_scaled, Y_big, random_state=0)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_shuffle, Y_shuffle, test_size=0.3, random_state=0)
# 定义偏最小二乘回归模型

svr = SVR(kernel='linear')
svr.fit(X_train, y_train)
# 输出最佳参数组合和交叉验证得分

# 预测测试集
y_pred = svr.predict(X_test)
# 计算R2分数
r2 = r2_score(y_test, y_pred)
print(f"PLS回归模型的R2分数为：{r2:.4f}")

