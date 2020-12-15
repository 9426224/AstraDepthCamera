import cv2
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

number = 0
cv2.namedWindow('depth')
cv2.namedWindow('color')

openni2.initialize("C://Users//9426224//PycharmProjects//DeepLearning//windows//SDK//x64//Redist")


dev = openni2.Device.open_any()
print(dev.get_device_info())
depth_stream = dev.create_depth_stream()

# 设置模式
video_mode = openni2.VideoMode()
depth_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                           resolutionX=1280,
                           resolutionY=1024,
                           fps=7))

depth_stream.set_mirroring_enabled(False)

depth_stream.start()

while True:

    depth_source = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(1024, 1280)
    img = np.uint8(depth_source.astype(float) / 37 - 16)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = 255 - img
    cv2.imshow('depth', img)


    ret, frame = cap.read()
    cv2.imshow('color', frame)
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        break
    if c == ord('s'):
        cv2.imwrite('depth/pic' + str(number) + '.jpg', img)
        cv2.imwrite('color/pic' + str(number) + '.jpg', frame)
        np.savetxt('txt/txt' + str(number) + '.txt', depth_source, fmt="%d", delimiter=" ")
        print('保存成功! 序号：'+str(number))
        number = number + 1

cv2.destroyAllWindows()
openni2.unload()
depth_stream.stop()
dev.close()
