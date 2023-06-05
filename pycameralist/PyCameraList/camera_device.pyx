from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "camera_device.hpp":
    cdef int listVideoDevices(vector[string] & list)
    cdef int listAudioDevices(vector[string] & list)
    cdef int test()


def list_video_devices():
    """
    列出所有可用的视频源。
    :return:
    """
    cdef int ret
    cdef vector[string] devices
    ret = listVideoDevices(devices)

    py_devices = []
    for i in range(devices.size()):
        py_devices.append((i, devices[i].decode('gbk')))
    return py_devices

def list_audio_devices():
    """
    列出所有可用的音频源。
    :return:
    """
    cdef int ret
    cdef vector[string] devices
    ret = listAudioDevices(devices)

    py_devices = []
    for i in range(devices.size()):
        py_devices.append((i, devices[i].decode('gbk')))
    return py_devices

def test_list_cameras():
    cdef int ret

    ret = test()
