import cv2
import time
import tensorflow as tf
import numpy as np
from PIL import Image

# url = 'rtsp://admin:TJDX8888@192.168.110.239:554'
url = 'rtsp://admin:123@192.168.110.174:8554/live'

cap = cv2.VideoCapture(url)

fps = 0.0

while True:
    t1 = time.time()

    ret, frame = cap.read()

    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video", frame)
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
