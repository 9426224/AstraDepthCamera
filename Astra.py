from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np

dist = "C://Users//9426224//PycharmProjects//DeepLearning//DepthCamera//windows//SDK//x64//Redist"
depth_stream = None
color_stream = None
depth_width = None
depth_height = None
depth_fps = None
color_width = None
color_height = None
color_fps = None
openni2.initialize(dist)


def init(w, h, f):
    global depth_width, depth_height, depth_fps, depth_stream, color_stream
    depth_width = w
    depth_height = h
    depth_fps = f

    dev = openni2.Device.open_any()

    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()

    depth_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                           resolutionX=depth_width,
                           resolutionY=depth_height,
                           fps=depth_fps))

    depth_stream.set_mirroring_enabled(False)

    color_stream.start()
    depth_stream.start()

    # dev.set_image_registration_mode(True)
    # dev.set_depth_color_sync_enabled(True)


def get_depth(begin, end):
    map_source = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_height,
                                                                                                          depth_width)
    map_clip = map_source[begin - 1:end - 1, :]
    return map_clip


def get_color():
    cframe = color_stream.read_frame()
    cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
    R = cframe_data[:, :, 0]
    G = cframe_data[:, :, 1]
    B = cframe_data[:, :, 2]
    cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
    return cframe_data
