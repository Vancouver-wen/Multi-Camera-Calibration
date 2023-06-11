```
# 源码安装 multical
pip install opencv_contrib_python==3.4.18.65 # 会报错版本不匹配，不要理会
pip install numpy==1.21
pip install easydict
pip install vtk==9.1.0
pip install pyvistaqt
pip install pyqt5
pip install colour
pip install qtawesome

# windows 需要进行额外的操作
# 安装 pycameralist
pip install pywin32
```


**TODO：**
1. 世界坐标系的调整
2. 将相机参数保存为标准json文件
3. TEST：读取标准json文件，将世界坐标轴可视化在每个相机图像中
4. 在linux下根据name和vid pid区分usb相机
5. 完成multical可视化，并记录需要的环境


## Use MyCalibration to calibrate multi-cameras
#### MyCalibration功能简介
1. 生成校准板
2. 多视角拍摄校准图片
3. 计算各视角相机参数（内参、外参、畸变系数）


#### MyCalibration的调用
```shell
pip install -r requirements.txt
```
```python
from config import config
from MyCalibration.py import MyCalibrate
my_calibrate=MyCalibrate(config=config)
my_calibrate.start_calibrate()
```


## 环境准备
#### multicam的安装

网址： https://github.com/oliver-batchelor/multical

1. 使用源码安装
```shell
# 建议使用源码安装
cd multical
pip install -v -e . # -v 打印详细信息  -e 使用源码安装  . 执行该目录下的setup.py
```
2. 从PyPi中安装
```shell
pip install multical
```
#### multical 命令行用法演示
```shell
# multical 生成校准板 
multical boards --boards ./multical/example_boards/my_charuco_16x22.yaml --paper_size A0 --pixels_mm 10 --write my_boards
multical boards --boards ./img_collect/my_boards/my_charuco.yaml --paper_size A0 --pixels_mm 10 --write ./img_collect/my_boards
# multical 校准计算内参外参畸变系数
multical calibrate --boards multical/example_boards/my_charuco_16x22.yaml  --image_path img_collect/images
# multical 可视化
multical vis --workspace_file img_collect/images/calibration.pkl
```
#### Windows 用户
**pycameralist的安装**

网址： https://gitee.com/jiangbin2020/py-camera-list-project
```
温馨提示：
同一厂家出的同一型号摄像头，他的vid和pid是一样的，利用上述方法就不行，
可以在购买摄像头的时候跟客服说，让他们给每个摄像头不同的vidpid号，
```

1. 使用源码安装
```shell
cd pycameralist
pip install -r requirements.py
pip install -v -e .
```
2. 从PyPi中安装
```shell
# 只适用于 python3.6，作者没有提供其他python版本的whl文件
pip install pycameralist
```

#### Ubuntu 用户
```
无需额外安装 PyPi包
```


<font color=#008000 >Bug Record:</font>
```
Q: AttributeError: module 'numpy' has no attribute 'int'.
A: np.int 在 NumPy 1.20 中已弃用，在 NumPy 1.24 中已删除
```
```
Q: cv2.aruco.CharucoBoard_create not found in OpenCV 4.7.0
A: 降低版本 https://stackoverflow.com/questions/75085270/cv2-aruco-charucoboard-create-not-found-in-opencv-4-7-0
This is due to a change that happened in release 4.7.0, when the Aruco code was moved from contrib to the main repository.
```
```
Q: TypeError: load() missing 1 required positional argument: 'Loader'
A: pyYAML版本过高 pip install pyyaml==5.4.1
```
```
Q: error: Microsoft Visual C++ 14.0 or greater is required. 
Get it with "Microsoft C++ Build Tools": 
https://visualstudio.microsoft.com/visual-cpp-build-tools/
A: 根据提示的网址安装 visual studio
```
```
Q: ERROR: Could not find a version that satisfies the requirement pyvista-qt (from versions: none)
ERROR: No matching distribution found for pyvista-qt
A: pip install pyvistaqt
```
```
Q: ERROR - No module named 'colour'
A: pip install colour
```
```
Q: ERROR - No module named 'qtawesome'
A: pip install qtawesome
```
