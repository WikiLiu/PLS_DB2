# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import cm
# from matplotlib import rcParams
#
# # 设置西文字体为新罗马字体
# from matplotlib import rcParams
# config = {
#     "font.family":'Times New Roman',  # 设置字体类型
#     "font.size": 80,
# #     "mathtext.fontset":'stix',
# }
#
#
# # 定义数据
# data = [-0.24505, 0.11747, -0.40791, -0.69066, -0.23948, -0.52935, -0.55545, -0.51385,
#         -0.12158, -0.09718, -0.29579, -0.23679, 0.08189, -0.13487, -0.73986, -0.89794,
#         -0.33833, -0.21985, 0.02265, 0.04971, 0.26304, -0.23750, -0.27777, -0.52022,
#         -0.52245, -0.56208, -0.00961, 0.39408, -0.10132, -0.89097, 0.01519, 0.67459,
#         -0.22432, 0.16324, 0.34207, -0.18881, -0.13705, 0.26210, 0.47806, -0.32331,
#         0.02412, -0.15839, 0.05541, -0.13758, 1.00000]
#
#
# # 定义子图布局
# fig, axs = plt.subplots(nrows=3, figsize=(6, 8))
#
# # 分割数据
# data1 = data[:15]
# data2 = data[15:30]
# data3 = data[30:]
#
# norm = plt.Normalize(-1, 1)
# norm_values = norm(data)
# map_vir = cm.get_cmap(name='viridis')
# colors = map_vir(norm_values)
#
# # 绘制子图1
# axs[0].bar(range(0,15), data1, color=colors[:15])
# axs[0].set_xticks(range(15))
# labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
# axs[0].set_xticklabels(labels)
# axs[0].set_yticks(range(-1, 2))
#
# # axs[0].set_title('')
#
# # 绘制子图2
#
# # labels = ['16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
# axs[1].bar(['16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'], data2, color=colors[15:30])
# axs[1].set_yticks(range(-1, 2))
# axs[0].set_ylabel("Pearson")
# axs[1].set_ylabel("Pearson")
# # 绘制子图3
# axs[2].set_ylabel("Pearson")
# axs[2].bar(['31','32','33','34','35','36','37','38','39','40','41','42','43','44','45'], data3, color=colors[30:])
#
# axs[2].set_yticks(range(-1, 2))
#
#
# # 添加色卡
# fig.subplots_adjust(right=0.8)
# cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
# norm = plt.Normalize(min(data), max(data))
# sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
# sm.set_array([])
# fig.colorbar(sm, cax=cax)
# # 显示图形
#
# plt.savefig('feature_importance.png',dpi=600, bbox_inches='tight')
#
# plt.show()
#
#
#
#
#
#
#
#
# #
# # import matplotlib.pyplot as plt
# # from matplotlib import cm
# #
# #
# # def draw_bar(key_name, key_values):
# #         plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# #         plt.rcParams['axes.unicode_minus'] = False
# #
# #         # 标准柱状图的值
# #         def autolable(rects):
# #                 for rect in rects:
# #                         height = rect.get_height()
# #                         if height >= 0:
# #                                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.02, '%.3f' % height)
# #                         else:
# #                                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height - 0.06, '%.3f' % height)
# #                                 # 如果存在小于0的数值，则画0刻度横向直线
# #                                 plt.axhline(y=0, color='black')
# #
# #         # 归一化
# #         norm = plt.Normalize(-1, 1)
# #         norm_values = norm(key_values)
# #         map_vir = cm.get_cmap(name='inferno')
# #         colors = map_vir(norm_values)
# #         fig = plt.figure()  # 调用figure创建一个绘图对象
# #         plt.subplot(111)
# #         ax = plt.bar(key_name, key_values, width=0.5, color=colors, edgecolor='black')  # edgecolor边框颜色
# #
# #         sm = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
# #         sm.set_array([])
# #         plt.colorbar(sm)
# #         autolable(ax)
# #
# #         plt.show()
# #
# #
# # if __name__ == '__main__':
# #         # multi_corr()
# #         key_name = ['时长', '鹤位', '设定量', '发油量', '发油率', '时间', '月份', '日期', '损溢量', '温度', '密度']
# #         key_values = [0.1, 0.9, 1, 1, 0.4, 0.3, -0.1, -0.6, 0.2, 0.4, 0.5]
# #         draw_bar(key_name, key_values)





import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

import pandas as pd
from sklearn import cross_decomposition, model_selection, preprocessing

df = pd.read_csv("output220206104400.csv", index_col=0)
df = df[[col for col in df.columns if col != 'DELTA_THICK_7'] + ['DELTA_THICK_7']]
id_to_match = "220206104400"
matched_rows = df.query(f'STRIP_NO_6 == ' + id_to_match)
# error = df.query(f'DELTA_THICK_7 > ' + "0.1" + " | DELTA_THICK_7 < -0.1")
# df = df.drop(error.index)
df = pd.concat([df.drop(matched_rows.index), matched_rows])
data_big = df.drop(["STAND_NO_6","STAND_NO_7"],axis=1)
X = data_big.drop(["DELTA_THICK_7", "STRIP_NO_6"], axis=1)
# 导入数据并进行均值方差归一化
x= preprocessing.minmax_scale(X)

X = x.data  # 特征向量
y = np.zeros(69)
y[-1] = 1

# 将30维数据降至2维
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# 绘制散点图
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('Set1', 2))
plt.colorbar(ticks=range(3))
plt.title('TSNE visualization of Iris dataset')
plt.xlabel('TSNE component 1')
plt.ylabel('TSNE component 2')
plt.show()