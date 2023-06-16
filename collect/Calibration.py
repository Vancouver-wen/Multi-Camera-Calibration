import sys
import cv2
from cv2 import aruco
import numpy as np
import time
import threading
import datetime
import shutil
import os
import json
from easydict import EasyDict
import matplotlib.pyplot as plt 

from .Collect import MyCollect


class MyCalibrate():
    """
    调用 multical calibrate 计算相机内参和相对外参 \n
    计算 首个相机 的外参，转换其他相机的外参 \n
    使用 start_calibrate
    """
    def __init__(self,config):
        self.config=config
        
        self.yaml_path=None
        self.imgfolder_path=None
        self.pkl_path=None
        self.json_path=None
        self.real_json_path=None
        if config.recollect==True:
            self.prepare_for_calivration()
        else:
            self.test_calibration()

    def prepare_for_calivration(self,):
        """
        生成校准板，并展示、保存 \t
        拍摄 多视角校准图片，并保存
        """
        # 获取 yaml文件 img文件夹 的 相对地址
        my_collect=MyCollect(config=self.config)
        yaml_path,imgfolder_path=my_collect.collect()
        pkl_path = self.config.img_path+"images/calibration.pkl"
        json_path= self.config.img_path+"images/calibration.json"
        del my_collect
        # 将 相对地址 转化为 绝对地址
        self.yaml_path=os.path.abspath(yaml_path)
        self.imgfolder_path=os.path.abspath(imgfolder_path)
        self.pkl_path=os.path.abspath(pkl_path)
        self.json_path=os.path.abspath(json_path)
    
    def test_calibration(self,):
        """
        当 config.recollect==False 时触发 \t
        此时应该满足：已经生成校准板、已经拍摄多视角校准图片
        """
        # 赋值相对地址
        yaml_path = self.config.img_path + "my_boards/my_charuco.yaml"
        imgfolder_path = self.config.img_path + "images/"
        pkl_path = self.config.img_path+"images/calibration.pkl"
        json_path= self.config.img_path+"images/calibration.json"
        # 将 相对地址 转化为 绝对地址
        self.yaml_path=os.path.abspath(yaml_path)
        self.imgfolder_path=os.path.abspath(imgfolder_path)
        self.pkl_path=os.path.abspath(pkl_path)
        self.json_path=os.path.abspath(json_path)

    def start_calibrate(self,):
        """
        该方法返回一个 相机校准json 的绝对地址
        """
        self.start_intrinsic_calibrate() # 注释 以方便 debug
        rmat,tvec=self.start_extrinsic_calibrate()
        self.standardize_json(rmat,tvec)
        # 返回标准json文件的path
        return os.path.abspath(self.real_json_path)
        
    def standardize_json(self,rmat,tvec):
        """
        将自己采集的数据，修正为： CMU Panoptic 标准数据 \n
        T的单位 m、米
        """
        self.real_json_path=self.config.img_path+'real_calibration.json'
        # 加载 all cameras 的 相机外参，根据 rmat tvec 进行修正，得到真实相机外参
        with open(self.real_json_path, "r", encoding="utf-8") as f:
            cams_real_param = json.load(f)
        # transform_R = rmat
        # transform_T = tvec
        # print("=> transform_R:",transform_R)
        # print("=> transform_T:",transform_T)
        for index in range(self.config.cam_num):
            cam_real_param=cams_real_param[index]
            origin_R=np.asarray(cam_real_param["R"], dtype = np.float64)
            origin_T=np.asarray(cam_real_param["T"], dtype = np.float64)
            # 计算 all cameras 的 真实 相机外参
            # 使用markdown展示计算过程（ 从相对外参到绝对外参.md )
            new_R=np.matmul(origin_R,rmat)
            new_T=origin_T+np.matmul(origin_R,tvec)
            cam_real_param["R"]=new_R.tolist()
            cam_real_param["T"]=new_T.tolist()
            
        # cams_real_param=[cams_real_param[0]] # 只看 cam1
        print("=> 所有相机的真实参数:",cams_real_param)
        # 保存 all cameras 的 真实 相机外参
        with open(self.real_json_path,'w',encoding='utf8') as f:
            # ensure_ascii=False才能输入中文，否则是Unicode字符
            # indent=2 JSON数据的缩进，美观
            json.dump(cams_real_param,f,ensure_ascii=False,indent=2)

    def start_extrinsic_calibrate(self,):
        # 加载 all cameras 的 相机内参
        with open(self.json_path, "r", encoding="utf-8") as f:
            cams_param = json.load(f)
        cams_real_param=[]
        # 整理 json 数据结构
        for index in range(self.config.cam_num):
            cam_name="cam%01d"%(index+1)
            cam_real_param=dict()
            cam_real_param["resolution"]=tuple(cams_param["cameras"][cam_name]["image_size"])
            cam_real_param["K"]=cams_param["cameras"][cam_name]["K"]
            cam_real_param["dist"]=cams_param["cameras"][cam_name]["dist"]
            if index==0:
                cam_real_param["R"]=cams_param["camera_poses"][cam_name]["R"]
                cam_real_param["T"]=cams_param["camera_poses"][cam_name]["T"]
            else:
                cam_name=cam_name+"_to_cam1"
                cam_real_param["R"]=cams_param["camera_poses"][cam_name]["R"]
                cam_real_param["T"]=cams_param["camera_poses"][cam_name]["T"]
            cams_real_param.append(cam_real_param)
        # 计算 cam1的 真实 相机外参
        rmat,tvec=self.cam1_extrinsic_calibrate(cam1_param=cams_real_param[0])
        print("=> cv2.solvePnP 计算得到 cam1 相机外参\n   ",f"R:{list(rmat)}\n   ",f"T:{list(tvec)}")
        # 检查一个旋转矩阵是否有效
        def validate_rmat(R):
            """旋转矩阵的性质
            旋转矩阵的行列式 det(R)=1
            旋转矩阵的逆矩阵等于其转职矩阵 R.I=R.T
            旋转矩阵的行向量或列向量都是单位向量
            """
            # 得到该矩阵的转置
            Rt = np.transpose(R)
            # 旋转矩阵的一个性质是，相乘后为单位阵
            shouldBeIdentity = np.dot(Rt, R)
            # 构建一个三维单位阵
            I = np.identity(3, dtype = R.dtype)
            # 将单位阵和旋转矩阵相乘后的值做差
            n = np.linalg.norm(I - shouldBeIdentity)
            # 如果小于一个极小值，则表示该矩阵为旋转矩阵
            isR=n < 1e-6
            return isR
        assert validate_rmat(rmat),"=> rmat不是旋转矩阵！"
        # 保存 all cameras 的 真实 相机外参
        self.real_json_path=self.config.img_path+'real_calibration.json'
        with open(self.real_json_path,'w',encoding='utf8') as f:
            # ensure_ascii=False才能输入中文，否则是Unicode字符
            # indent=2 JSON数据的缩进，美观
            json.dump(cams_real_param,f,ensure_ascii=False,indent=2)
        # import pdb;pdb.set_trace()
        return rmat,tvec
        
    def cam1_extrinsic_calibrate(self,cam1_param):
        # 鼠标操作，鼠标选中源图像中需要替换的位置信息
        def mouse_action01(event, x, y, flags, param_array):
            cv2.imshow('collect img_before coordinate', img_before_copy)
            if event == cv2.EVENT_LBUTTONUP:
                # 画圆函数，参数分别表示原图、坐标、半径、颜色、线宽(若为-1表示填充)
                # 这个是为了圈出鼠标点击的点
                cv2.circle(img_before_copy, (x, y), 2, (0, 0, 255), -1)
                # 用鼠标单击事件来选择坐标
                # 将选中的四个点存放在集合中，在收集四个点时，四个点的点击顺序需要按照 img_src_coordinate 中的点的相对位置的前后顺序保持一致
                # print(f'{x}, {y}') # 在收集四个点的过程中打印 像素位置
                param_array.append([x, y])
        
        print("=> 请按照象限递增的顺序，逆时针选择水平桌面的四个点，选完四个点后，按 esc 退出 ..")
        # 获得来自 cam1 的一帧图像
        if self.config.recollect==True:
            cap= cv2.VideoCapture(0)
            ret, frame = cap.read() 
            cap.release()
            assert ret,"ERROR: can not get view from cam1"
            img_before=frame
        else:
            img_before = cv2.imread('./img_collect/images/cam1/image02.jpg', cv2.IMREAD_COLOR)
        img_before_copy = np.tile(img_before, 1)
        before_coordinate = []
        cv2.namedWindow('collect img_before coordinate',0)
        cv2.setMouseCallback('collect img_before coordinate', mouse_action01, before_coordinate)
        while True:
            # 按esc 或 采集满4个点  退出鼠标采集行为
            if cv2.waitKey(1) == 27 or len(before_coordinate)==4: 
                cv2.destroyAllWindows()
                break
        print("=> 依次选取的像素点位置为：",before_coordinate)
        imgPoints = np.asarray(before_coordinate, dtype = np.float64)
        def get_objPoints(horizontal_table):
            width,length,height=horizontal_table
            objPoints=[
                [int(width/2),int(length/2),height], # 第一象限
                [-int(width/2),int(length/2),height], # 第二象限
                [-int(width/2),-int(length/2),height], # 第三象限
                [int(width/2),-int(length/2),height] # 第四象限
            ]
            return objPoints
        # 四个点的世界坐标系坐标
        objPoints = np.asarray(get_objPoints(self.config.horizontal_table), dtype = np.float64)
        # 相机内参、畸变
        cameraMatrix = np.asarray( cam1_param["K"] ,dtype = np.float64)
        distCoeffs = np.asarray( cam1_param["dist"] , dtype = np.float64)
        # pnp算法
        retval, rvec, tvec = cv2.solvePnP(
            objPoints, imgPoints, cameraMatrix, distCoeffs,
            flags=cv2.SOLVEPNP_AP3P,
        )
        tvec=tvec/1000 # 把单位从 mm 转换为 m
        """flags opencv计算pnp的算法
        SOLVEPNP_ITERATIVE = 0,
        SOLVEPNP_EPNP      = 1, //!< EPnP: Efficient Perspective-n-Point Camera Pose Estimation @cite lepetit2009epnp
        SOLVEPNP_P3P       = 2, //!< Complete Solution Classification for the Perspective-Three-Point Problem 
        SOLVEPNP_DLS       = 3, //!< A Direct Least-Squares (DLS) Method for PnP  @cite hesch2011direct
        SOLVEPNP_UPNP      = 4, //!< Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation 
        SOLVEPNP_AP3P      = 5, //!< An Efficient Algebraic Solution to the Perspective-Three-Point Problem 
        SOLVEPNP_MAX_COUNT      //!< Used for count
        """
        """cv2.solvePnP 输入
        objPoints：N*3或者3*N的世界坐标系中的3D点坐标，单位mm
        imagePoints：N*2或者2*N的图像坐标系中点的坐标，单位pixel
        cameraMatrix,distCoeffs：相机内参矩阵 畸变系数
        """
        """cv2.solvePnP 返回值
        retval:true 或 false
        rvec：旋转向量，以Rodrigues向量表示，3x1
        tvec：平移向量，3x1
        处理矩阵三维转换时，通常采用旋转矩阵，但是旋转变换其实只有三个自由度，用旋转向量表达时更为简洁
        旋转向量和旋转矩阵之间可以通过罗德里格斯公式进行转换
        R = cv2.Rodrigues(a)：a为旋转向量，R为旋转矩阵
        """
        rmat=cv2.Rodrigues(rvec)[0]
        tvec=np.squeeze(tvec)
        return rmat,tvec

    def start_intrinsic_calibrate(self,):
        command = " multical calibrate "
        command = command + " --boards " + self.yaml_path
        command = command + " --image_path " + self.imgfolder_path
        process, exitcode=self.shell_command(command)
        print("=> 可视化 intrinsic calibration 的结果 ..")
        command = " multical vis "
        command = command + " --workspace_file " + self.pkl_path
        process, exitcode=self.shell_command(command)

    def shell_command(self,command):
        """
        执行 shell 命令并实时打印输出
        """
        from subprocess import Popen, PIPE, STDOUT
        print("=> 执行shell命令:",command)
        process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
        with process.stdout:
            for line in iter(process.stdout.readline, b''): # 沿迭代器循环，直到 b'' 这应该是规定好的
                print(line)
        exitcode = process.wait()
        return process, exitcode