import threading

from scipy import signal

from Astra import *
import cv2
import matplotlib.pyplot as plt

depth_source = None
color_source = None
depth_img = None
color_img = None

number = 0


# capture_width = 1280
# capture_height = 720
#
# cap = cv2.VideoCapture(0)
# cap.set(3, capture_width)
# cap.set(4, capture_height)


# 鼠标点击抓取坐标
def mouseCallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(y, x, depth_source[y, x])


# 读取深度流
def read():
    global depth_source, color_source
    while True:
        # depth_source = get_depth(1, 480)
        depth_source = get_depth(1, 1024)
        # color_source = get_color()
        # depth_source = get_depth(384, 576)


def depth():
    global depth_img, number
    # cv2.namedWindow('color')
    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth', mouseCallback)
    while True:
        while depth_source is not None:
            # while depth_source is not None and cap.isOpened():

            # q = calc(depth_source)

            # ret, imgc = cap.read()
            # img = np.uint8(depth_source.astype(float) / 37 - 16)
            img = np.uint8(depth_source.astype(float) / 45.882352941176470588235294117647 - 6.5384615384615384615384615384619)
            # img = np.uint8(data.astype(float) * 255 / 2 ** 12 - 1)

            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # 膨胀与腐蚀/开闭
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            iterations = 10
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations)

            # 二维中值滤波
            img = s

            data = np.uint16((img.astype(float) + 6.5384615384615384615384615384619) * 45.882352941176470588235294117647)

            calc(data)

            img = 255 - img

            cv2.imshow('depth', img)
            # cv2.imshow('color', imgc)
            c = cv2.waitKey(1) & 0xff
            if c == 27:
                cv2.destroyAllWindows()
                depth_stream.stop()
                openni2.unload()
                # cap.release()
                break
            # if c == ord('s'):
            #     cv2.imwrite('img/depth_picture' + str(number) + '.jpg', img)
            #     cv2.imwrite('img/color_picture' + str(number) + '.jpg', imgc)
            #     np.savetxt('data_picture'+str(number), )
            #     print('保存成功:' + 'picture' + str(number) + '.jpg')
            #     number = number + 1


def calc(mat):
    result = np.zeros(1280)

    # for i in range(mat.shape[1]):
    #     for j in range(mat.shape[0]):
    #         if result[i] != 0 and mat[j][i] != 0:
    #             result[i] = min(result[i], mat[j][i])
    #         elif mat[j][i] != 0:
    #             result[i] = mat[j][i]

    for i in range(mat.shape[1]):
        for j in range(mat.shape[0]):
            if mat[j][i] <= 400 or mat[j][i] >= 10000:
                mat[j][i] = 0
            if result[i] != 0 and mat[j][i] != 0:
                result[i] = min(result[i], mat[j][i])
            elif mat[j][i] != 0:
                result[i] = mat[j][i]

    plt.plot(result)
    plt.show()



if __name__ == '__main__':
    init(1280, 1024, 7)
    # init(640, 480, 30)
    p1 = threading.Thread(target=read)
    p2 = threading.Thread(target=depth)
    p1.start()
    p2.start()
