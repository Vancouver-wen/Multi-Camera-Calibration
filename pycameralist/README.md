# PyCameraList

缘起

python下使用opencv库使用VideoCapture调用摄像头时，只能传入枚举号，而不知道
相应的相机名称。电脑有多个摄像头时，选择摄像头就会很不方便；电脑只有一个摄像头
时有时摄像头的ID也不一定是0。我们希望能够获取所有CV可以使用的摄像头信号源。
这个库的目的就是为解决这个问题而生。


## 安装
```cmd
pip install pycameralist
```

## 使用

```python
from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices

cameras = list_video_devices()
print(dict(cameras))
#return: {0: 'Intel(R) RealSense(TM) 3D Camera (Front F200) RGB', 1: 'NewTek NDI Video', 2: 'Intel(R) RealSense(TM) 3D Camera Virtual Driver', 3: 'Intel(R) RealSense(TM) 3D Camera (Front F200) Depth', 4: 'OBS-Camera', 5: 'OBS-Camera2', 6: 'OBS-Camera3', 7: 'OBS-Camera4', 8: 'OBS Virtual Camera'}
 
audios = list_audio_devices()
print(dict(audios))
#return:  {0: '麦克风阵列 (Creative VF0800)', 1: 'OBS-Audio', 2: '线路 (NewTek NDI Audio)'}


```



