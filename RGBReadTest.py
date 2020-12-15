from openni import openni2
from openni.openni2 import c_api
import numpy as np
import cv2
#
# depth_width = 1280
# depth_height = 1024
# depth_fps = 7

depth_width = 640
depth_height = 480
depth_fps = 30


def mousecallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(y, x, dpt[y, x])


if __name__ == "__main__":
    openni2.initialize()
    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                   resolutionX=depth_width, resolutionY=depth_height, fps=depth_fps))
    depth_stream.set_mirroring_enabled(False)
    depth_stream.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth', mousecallback)
    number = 0
    while True:

        frame = depth_stream.read_frame()
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([depth_height, depth_width, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt2 = cv2.convertScaleAbs(dpt, alpha=(255 / np.max(dpt)))

        # print(dpt2.shape[:])
        # print(dpt2.)
        dpt2 = np.reshape(dpt2, (depth_height, depth_width, 1))
        # cv2.imshow('depth', dpt2)
        # dpt3 = cv2.applyColorMap(dpt2, cv2.COLORMAP_RAINBOW)
        # for height in range(dpt3.shape[0]):
        #     for width in range(dpt3.shape[1]):
        #         if (dpt3[height, width] == [0, 0, 255]).all():
        #             dpt3[height, width] = [0, 0, 0]
        # dpt3 = cv2.applyColorMap(dpt2, cv2.COLORMAP_OCEAN)
        cv2.imshow('depth', dpt2)
        ret, frame = cap.read()
        cv2.imshow('color', frame)

        key = cv2.waitKey(1)
        if int(key) == 27:
            break
        if key == ord('s'):
            cv2.imwrite('E:/Data/depth_pic/gray/picture' + str(number) + '.jpg', dpt2)
            print('保存成功' + 'picture' + str(number) + '.jpg')
            number = number + 1
    depth_stream.stop()
    dev.close()