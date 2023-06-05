# Camera-Calibration

网址： https://github.com/oliver-batchelor/multical

Bug Record:
```
Q: AttributeError: module 'numpy' has no attribute 'int'.
A: np.int 在 NumPy 1.20 中已弃用，在 NumPy 1.24 中已删除
Q: cv2.aruco.CharucoBoard_create not found in OpenCV 4.7.0
A: 降低版本 https://stackoverflow.com/questions/75085270/cv2-aruco-charucoboard-create-not-found-in-opencv-4-7-0
This is due to a change that happened in release 4.7.0, when the Aruco code was moved from contrib to the main repository.
Q: TypeError: load() missing 1 required positional argument: 'Loader'
A: pyYAML版本过高 pip install pyyaml==5.4.1
Q: error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
A: 根据提示的网址安装 visual studio
```

## multicam的安装
1. 使用源码安装
```python
cd multical
pip install -v -e . # -v 打印详细信息  -e 使用源码安装  . 执行该目录下的setup.py
```
2. 从PyPi中安装
```python
pip install multical
```
## pycameralist的安装

pycameralist能够获取所有相机的 id 与 name
但区分不同摄像头的关键应该是 PID 和VID，而不是name，尤其是当相机型号完全相同的时候
https://blog.csdn.net/qq_41043389/article/details/124664485
```
温馨提示：同一厂家出的同一型号摄像头，他的vid和pid是一样的，利用上述方法就不得行，所以你可以在购买摄像头的时候跟客服说，让他们给每个摄像头不同的vidpid号，我经常在淘宝买摄像头，他们可以提供这个服务。
```
```

https://gitee.com/jiangbin2020/py-camera-list-project
```
https://github.com/pvys/CV-camera-finder

1. 使用源码安装
```python
cd pycameralist
pip install -r requirements.py
pip install -v -e .
```
2. 从PyPi中安装
```python
# 只适用于 python3.6，作者没有提供其他python版本的whl文件
pip install pycameralist
```

## multicam生成校准板
```python
multical boards --boards ./multical/example_boards/my_charuco_16x22.yaml --paper_size A0 --pixels_mm 10 --write my_boards

# In my case, we use boards:
# charuco_16x22 CharucoBoard {type='charuco', aruco_dict='4X4_1000', aruco_offset=0, size=(16, 22), marker_length=0.0375, square_length=0.05, aruco_params={}}
# Wrote my_images/charuco_16x22.png
```



## 自定义程序引导的校准图像收集
```python
from MyCollect import MyCollect
my_collect=MyCollect(cam_num=1)
my_collect.collect(img_path="./images/")
# 开启一个窗口，按 s 保存
```

## multicam计算内参外参畸变系数

## 世界坐标系的调整

## 将相机参数保存为标准json文件

## TEST：读取标准json文件，将世界坐标轴可视化在每个相机图像中