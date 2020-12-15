import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread(r"C:\Users\9426224\Desktop\Data\SGBM\L.JPG")
imgR = cv2.imread(r"C:\Users\9426224\Desktop\Data\SGBM\R.JPG")

window_size = 3

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=240,
    blockSize=3,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# a = stereo.compute(imgL, imgR)

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
plt.figure(dpi=500)
plt.imshow(disparity, 'gray')
plt.show()
# cv2.imshow("1", disparity)
