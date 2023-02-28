import numpy as np
from sklearn.decomposition import PCA

# 生成数据，X为特征矩阵，y为目标变量
X = np.random.rand(100, 30)
y = np.random.rand(100,)

# 将X标准化
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 创建PCA对象，指定要保留的主成分数为20
pca = PCA(n_components=20)

# 对X进行PCA降维
X_pca = pca.fit_transform(X)

# 利用PCA重构回归
X_reconstructed = pca.inverse_transform(X_pca)

# 计算重构误差
mse = ((X - X_reconstructed) ** 2).mean(axis=None)

# 输出重构误差
print('重构误差为：', mse)
