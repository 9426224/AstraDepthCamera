import threading
import time
import numpy as np
import cv2


# 104
# mtx = np.array([[2.64939091e+03, 0.00000000e+00, 9.30920581e+02],
#                 [0.00000000e+00, 2.63103129e+03, 4.49088683e+02],
#                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#
# dist = np.array([[-3.32133697e-01, -3.02928247e+00, 8.49628077e-03, -5.58582968e-04, 1.45505982e+01]])

# 239
mtx = np.array([[1.54089335e+03, 0.00000000e+00, 1.28523207e+03],
                [0.00000000e+00, 1.52797743e+03, 7.17899962e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[-0.41121255, 0.21959859, 0.00073101, 0.0011966, -0.06909926]])

# mtx, dist = CameraCalibration.Calibration()

# rtsp串流的地址
rtspUrl = "rtsp://admin:TJDX8888@192.168.110.239:554"
# 捕获rtsp格式的视频流
cap = cv2.VideoCapture(rtspUrl)
# 设置帧缓存区
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
# q = queue.LifoQueue()
p, q = cap.read()


def Read():
    global q
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        q = frame


def Video():
    global t
    while cap.isOpened():
        frame = q

        t = time.time()

        frame = cv2.undistort(frame, mtx, dist, None, mtx)

        print(time.time() - t)

        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        cv2.imshow("video", frame)

        # 按下Esc中断串流
        key = cv2.waitKey(1) & 0xff
        if key == 27:  # 27 is the Esc Key
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=Read)
    p2 = threading.Thread(target=Video)
    p1.start()
    p2.start()
