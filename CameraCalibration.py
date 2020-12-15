import glob
import cv2
import numpy as np


def Calibration():
    w = 13  # 长边个数
    h = 9  # 短边个数

    objp = np.zeros((w * h, 3), np.float32)  # 创建0矩阵,w*h行3列,存放角点世界坐标
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    # 储存棋盘格角点的世界坐标和图像坐标对
    objp_list = []  # 在世界坐标系中的三维点
    corners_list = []  # 在图像平面的二维点

    images = glob.glob("D:/Work File/CameraCalibration/104/*.jpg")

    for frame in images:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，共w×h个点，gray必须是8位灰度或者彩色图，（w,h）为角点规模
        ret, corners = cv2.findChessboardCorners(gray, (w, h))

        # 如果找到足够点对，将其存储起来
        if ret:
            objp_list.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))

            if [corners2]:
                corners_list.append(corners2)
            else:
                corners_list.append(corners)

            # 绘制出寻找的棋盘点格
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(2000)

    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, img_size, None, None)

    print("ret:", ret)  # ret为bool值
    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数 distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量，外参数
    print("tvecs:\n", tvecs)  # 平移向量，外参数

    return mtx, dist


if __name__ == '__main__':
    Calibration()
