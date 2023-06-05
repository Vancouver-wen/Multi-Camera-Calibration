#!/usr/bin/env python
# coding=utf-8

from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices

cameras = list_video_devices()
print(dict(cameras))

audios = list_audio_devices()
print(dict(audios))