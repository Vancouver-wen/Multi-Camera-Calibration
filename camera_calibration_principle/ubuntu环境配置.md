**multicam的安装**

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

# multical 命令行用法演示
# multical 生成校准板 
multical boards --boards ./multical/example_boards/my_charuco_16x22.yaml --paper_size A0 --pixels_mm 10 --write my_boards
multical boards --boards ./img_collect/my_boards/my_charuco.yaml --paper_size A0 --pixels_mm 10 --write ./img_collect/my_boards
# multical 校准计算内参外参畸变系数
multical calibrate --boards multical/example_boards/my_charuco_16x22.yaml  --image_path img_collect/images
# multical 可视化
multical vis --workspace_file img_collect/images/calibration.pkl
```

**linux_requirements的安装**
1. 安装multical的附加环境
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

```
2. 安装Camera-Calibration的环境
```
cached_property==1.5.2
colour==0.1.5
Cython==0.29.35
easydict==1.10
matplotlib==3.3.4
natsort==8.3.1
numpy==1.19.5
numpy_quaternion==2022.4.3
omegaconf==2.3.0
opencv_contrib_python==3.4.18.65
packaging==23.1
palettable==3.3.3
py_structs==0.2.14
pyserial==3.5
pyvista==0.39.1
pyvistaqt==0.2.0
pywin32==306
PyYAML==5.4.1
QtAwesome==1.2.3
QtPy==2.3.1
scipy==1.10.1
setuptools==66.0.0
simple_parsing==0.1.2.post1
tqdm==4.65.0
vtk==9.0.2
```
