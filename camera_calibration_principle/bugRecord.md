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
