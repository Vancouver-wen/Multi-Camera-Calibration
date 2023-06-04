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

```

# multicam的安装
1. 使用源码安装
```python
cd multical
pip install -v -e . # -v 打印详细信息  -e 使用源码安装  . 执行该目录下的setup.py
```
2. 从PyPi中安装
```python
pip install multical
```

# multicam生成校准板
```python
multical boards --boards ./multical/example_boards/my_charuco_16x22.yaml --paper_size A0 --pixels_mm 10 --write my_boards

# In my case, we use boards:
# charuco_16x22 CharucoBoard {type='charuco', aruco_dict='4X4_1000', aruco_offset=0, size=(16, 22), marker_length=0.0375, square_length=0.05, aruco_params={}}
# Wrote my_images/charuco_16x22.png
```

# 自定义程序引导的校准图像收集
```python
from MyCollect import MyCollect
my_collect=MyCollect(cam_num=1)
my_collect.collect(img_path="./images/")
# 开启一个窗口，按 s 保存
```

# multicam计算内参外参畸变系数

# 世界坐标系的调整

# 将相机参数保存为标准json文件

# TEST：读取标准json文件，将世界坐标轴可视化在每个相机图像中