from sklearn.cross_decomposition import PLSRegression

# 创建PLSRegression对象
pls = PLSRegression(n_components=2)

# 用大类数据集训练PLS模型
pls.fit(data_big.iloc[:, :-1], data_big.iloc[:, -1])

# 获取模型的系数和截距
coefficients = pls.coef_
intercept = pls.y_mean_ - np.dot(pls.x_mean_, coefficients)


# 对小类数据集进行预测
predictions = np.dot(data_small.iloc[:, :-1], coefficients) + intercept

# 计算预测误差
errors = data_small.iloc[:, -1] - predictions

# 使用误差对模型参数进行纠偏
coefficients_corrected = coefficients + np.dot(data_small.iloc[:, :-1].T, errors) / np.linalg.norm(errors)
intercept_corrected = intercept + np.mean(errors)
# 使用纠偏后的模型参数对小类数据集进行重新预测
predictions_corrected = np.dot(data_small.iloc[:, :-1], coefficients_corrected) + intercept_corrected






















'''
from sklearn.cross_decomposition import PLSRegression

# 创建PLSRegression对象
pls = PLSRegression(n_components=2)

# 用大类数据集训练PLS模型
pls.fit(data_big.iloc[:, :-1], data_big.iloc[:, -1])

# 获取模型的系数和截距
coefficients = pls.coef_
intercept = pls.y_mean_ - np.dot(pls.x_mean_, coefficients)
# 对小类数据集进行预测
predictions = np.dot(data_small.iloc[:, :-1], coefficients) + intercept
# 计算预测误差
errors = data_small.iloc[:, -1] - predictions

# 使用小类数据集对模型参数进行拟合
pls_final = PLSRegression(n_components=2)
pls_final.fit(data_small.iloc[:, :-1], data_small.iloc[:, -1])
coefficients_final = pls_final.coef_
intercept_final = pls_final.y_mean_ - np.dot(pls_final.x_mean_, coefficients_final) + np.mean(errors)
# 使用最终的模型参数对数据进行预测
predictions_final = np.dot(data_test.iloc[:, :-1], coefficients_final) + intercept_final
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# 将大类数据集分为训练集和验证集
train_size = 0.8
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
n_train = int(train_size * len(data_big))
train_indices = list(kf.split(data_big))[0][0][:n_train]
X_train = data_big.iloc[train_indices, :-1]
y_train = data_big.iloc[train_indices, -1]
X_val = data_big.iloc[:, :-1].drop(train_indices)
y_val = data_big.iloc[:, -1].drop(train_indices)

# 选择潜变量数量
n_components_list = [1, 2, 3, 4, 5]
best_n_components = 0
best_mse = float('inf')
for n_components in n_components_list:
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_val)
    mse = np.mean((y_val - y_pred) ** 2)
    if mse < best_mse:
        best_mse = mse
        best_n_components = n_components
# 用最优的潜变量数量重新拟合模型
pls_final = PLSRegression(n_components=best_n_components)
pls_final.fit(data_big.iloc[:, :-1], data_big.iloc[:, -1])

# 使用小类数据集对模型进行拟合
coefficients_final = pls_final.coef_
intercept_final = pls_final.y_mean_ - np.dot(pls_final.x_mean_, coefficients_final)
predictions_final = np.dot(data_small.iloc[:, :-1], coefficients_final) + intercept_final


'''