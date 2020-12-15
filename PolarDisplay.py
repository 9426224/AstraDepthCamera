import math
import time
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth, MeanShift


# 加载
# init()

# 1280像素对应58.4度平面视场角计算得出的单像素对应的角度大小
CameraHorizontalAngle = 58.4
HorizontalResolution = 1280

pixel = CameraHorizontalAngle / HorizontalResolution

# 原始文本路径
# path = r"C:\Users\9426224\Desktop\Data\text"
path = r"C:\Users\9426224\Desktop\Data\text\text50.txt"


# 读取文本到mat矩阵
# mat = get_depth(1, 1024)
# mat = np.loadtxt(r"C:\Users\9426224\Desktop\Data\text\text" + str(number) + ".txt")
mat = np.loadtxt(path)


# ---------------------------------------------------------------------------------------------------------------------------
# # 处理原始文件，转换300-12000mm的深度信息至0-255深度的灰度图用于显示
# img = np.uint8(mat.astype(float) / 45.882352941176470588235294117647 - 6.5384615384615384615384615384619)
# # 反转图像黑白
# img = 255 - img
# # 输出图像
# cv2.imshow('3', img)
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 生成一维矩阵，长度与输出图像保持一致
result = np.zeros(mat.shape[1])

# 处理矩阵，将二维矩阵最近信息点提取到一维矩阵中
mat = mat.T
mat[mat == 0] = 12000
j = 0
for i in mat:
    result[j] = i[np.argmin(i)]
    j = j + 1
result[result == 12000] = 0
# # 计算矩阵处理所需时间时间
# t1 = time.time()
# # 处理矩阵，将二维矩阵最近信息点提取到一维矩阵中
# for i in range(mat.shape[1]):
#     for j in range(mat.shape[0]):
#         if result[i] != 0 and mat[j][i] != 0:
#             result[i] = min(result[i], mat[j][i])
#         elif mat[j][i] != 0:
#             result[i] = mat[j][i]
# print("旧方式读取数组", time.time() - t1)

# 横向遍历数组
# t2 = time.time()
#
# for i in range(mat.shape[0]):
#     for j in range(mat.shape[1]):
#         if result2[j] != 0 and mat[i][j] != 0:
#             result2[j] = min(result2[j], mat[i][j])
#         elif mat[i][j] != 0:
#             result2[j] = mat[i][j]
#
# print(time.time() - t2)

# theta为散点的角度信息转为弧度显示，angle为散点的角度信息(0度为分界线)
theta = np.zeros(result.shape[0])
angle = np.zeros(result.shape[0])


# 计算theta与angle的值
for i in range(result.shape[0]):
    if result[i] is not 0:
        angle[i] = (i - 640) * pixel
        theta[i] = math.radians((i - 640) * pixel + 90)

# 设置plt输出图像的DPI
plt.figure(dpi=500)
# 生成一维矩阵点信息
plt.plot(result)
# 一维矩阵数据图片显示
plt.show()
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 使用theta数据进行散点上色
colors = theta
# 设置散点绘制的大小
area = 1
# 设置plt输出图像的DPI
plt.figure(dpi=500)
# 生成极坐标信息
ax = plt.subplot(111, projection='polar')
# 设置极坐标为扇形并翻转极坐标显示方向
ax.set_theta_direction(-1)
ax.set_theta_zero_location('W')
ax.set_thetalim(np.pi / 4, np.pi * 3 / 4)
# 散点绘制
c = ax.scatter(theta, result, c=colors, s=area, cmap='hsv', alpha=0.75)
# 极坐标图片显示
plt.show()
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# MeanShift方式聚类
t1 = time.time()

# 带宽，也就是以某个点为核心时的搜索半径 数据集:dataset_X样本 quantile:分位数 n_samples:使用样本大小
bandwidth = estimate_bandwidth(result.reshape(-1, 1), quantile=0.05, n_samples=1280)
# 设置均值偏移函数
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(result.reshape(-1, 1))
print(ms.cluster_centers_)
labels = ms.labels_

print(time.time() - t1)

# 设置plt输出图像的DPI
plt.figure(dpi=500)
# 生成聚类信息
plt.plot(labels)
# 聚类信息图片显示
plt.show()

# KMeans方式聚类
# km = KMeans(n_clusters=4)
# km.fit(result.reshape(-1,1))
# print(km.cluster_centers_)
# labels = km.labels_
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 存储输出的深度信息
depth_return = np.zeros(100)
# 存储输出的角度信息
angle_return = np.zeros(100)
# 存储用于显示极坐标的输出角度信息的弧度
theta_return = np.zeros(100)

# 计算输出的信息取舍
for i in range(labels.shape[0]):
    if depth_return[labels[i]] != 0:
        depth_return[labels[i]] = min(result[i], depth_return[labels[i]])
        if result[i] >= depth_return[labels[i]]:
            angle_return[labels[i]] = angle[i]
            theta_return[labels[i]] = theta[i]
    else:
        depth_return[labels[i]] = result[i]
        angle_return[labels[i]] = angle[i]
        theta_return[labels[i]] = theta[i]

# 使用theta数据进行散点上色
colors = theta_return
# 设置散点绘制的大小
area = 1
# 设置plt输出图像的DPI
plt.figure(dpi=500)
# 生成极坐标信息
ax = plt.subplot(111, projection='polar')
# 设置极坐标为扇形并翻转极坐标显示方向
ax.set_theta_direction(-1)
ax.set_theta_zero_location('W')
ax.set_thetalim(np.pi / 4, np.pi * 3 / 4)
# 散点绘制
c = ax.scatter(theta_return, depth_return, c=colors, s=area, cmap='hsv', alpha=0.75)
# 极坐标图片显示
plt.show()


# 等待cv2显示的灰度图
# cv2.waitKey()