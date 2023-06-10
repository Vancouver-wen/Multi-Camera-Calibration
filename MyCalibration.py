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

from config import config

class MyBoardDetector():
    '''
    实例化(__init__)时，根据config文件创建 Calibration Pattern \n
    被调用(detect)时，输入一个图像，输入表示该图像能够用于 Calibration
    '''
    def __init__(self,config):
        from cv2 import aruco
        # ChAruco board variables
        self.CHARUCOBOARD_ROWCOUNT = config.board.size[0]
        self.CHARUCOBOARD_COLCOUNT = config.board.size[1]
        self.SQUARE_LENGTH = config.board.square_length
        self.MARKER_LENGTH = config.board.marker_length
        self.ARUCO_DICT = aruco.getPredefinedDictionary(dict=cv2.aruco.DICT_4X4_1000)
        # Create constants to be passed into OpenCV and Aruco methods
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=self.CHARUCOBOARD_ROWCOUNT ,
            squaresY=self.CHARUCOBOARD_COLCOUNT,
            squareLength=self.SQUARE_LENGTH,
            markerLength=self.MARKER_LENGTH,
            dictionary=self.ARUCO_DICT,
        )
        # Set response_threshold
        self.response_threshold=config.response_threshold
    def detect(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=self.ARUCO_DICT)
        if corners == ():
            # print("=> 没检测到charuco board,detect no corners ..")
            return False
        response,_, _ = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=self.CHARUCO_BOARD
        )
        return response

class MyThread(threading.Thread):
    """
    被 MyThreads 类调用
    """
    def __init__(self, cap,my_board_detector,config,):
        super(MyThread, self).__init__() 
        self.cap=cap
        self.my_board_detector=my_board_detector
        self.config=config
        self.width,self.height=self.config.resolution
        self.frame=None
        self.rec=True # 能够成功检测到校准板
        self.response=0 # 检测到的角点数量
    
    def get_real_frame(self):
        _,frame = self.cap.read()
        # 判断 frame 是否为 None
        if frame is None: # print(type(frame))
            print("出现带宽限制！",type(frame))
            frame = np.full((self.width, self.height, 3), 255, dtype = np.uint8) # 填充空白，让程序能够继续运行
        else:
            frame=cv2.resize(frame,dsize=(self.width, self.height))
        # 判断 运行 平台
        import platform
        plat = platform.system().lower()
        if plat == 'windows':
            frame=frame
        elif plat == 'linux':
            frame = cv2.flip(frame, 1) # 在linux系统中，opencv获取的图像是 镜像 的，需要 flip
        return frame

    def run(self):
        self.frame=self.get_real_frame()
        self.response=self.my_board_detector.detect(self.frame)
        self.rec=True if self.response>=self.config.response_threshold else False
        
    def get_result(self):
        return self.frame,self.rec,self.response
    
class MyThreads():
    """
    实例化(__init__)时，初始化所有相机对象，根据 config文件 初始化Calibration Pattern \n
    运行(get_frames_recs)时，返回所有相机同步拍摄的图片，以及每个图片是否能够作为"校准图片" \n
    captures=[{"rec","frame"},] 是 get_frames_recs 的返回值 
    """
    def __init__(self,config):
        self.config=config
        self.cam_num=config.cam_num
        self.resolution=config.resolution
        # 初始化 所有 相机对象
        self.caps=self.get_caps(self.resolution)
        # 初始化 校准板角点检测器
        self.my_board_detector=MyBoardDetector(config)

    def get_real_cap(self,cam_index):
        import platform
        plat = platform.system().lower()
        if plat == 'windows':
            cap = cv2.VideoCapture(cam_index,cv2.CAP_DSHOW) 
        elif plat == 'linux':
            cap = cv2.VideoCapture(cam_index*2)
        return cap
    
    def get_caps_name_id(self,):
        # 使用PyCameralist打印所有相机的 "虚拟编号"和"name" 
        from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices
        cameras = list_video_devices()
        for index,camera in enumerate(cameras):
            cameras[index]=camera[1]
        # 在 初始化每个相机的时候， 和 "vid pid", 涉及操作系统底层，分 windows 和 linux 两种情形
        import platform
        plat = platform.system().lower()
        if plat == 'windows':
            # 获取所有 USB 设备的 VID PID
            import win32com.client # pip install pypiwin32
            wmi = win32com.client.GetObject ("winmgmts:")
            usb_ids=[]
            for usb in wmi.InstancesOf ("Win32_USBHub"):
                if "&PID" in usb.DeviceID:
                    usb_id=usb.DeviceID.split('\\')[1]
                    VID,PID=usb_id.split('&')[0].split('_')[1],usb_id.split('&')[1].split('_')[1]
                    usb_ids.append({"VID":VID,"PID":PID})
            # 对所有的 USB 设备 VID PID 进行轮询(采用了dll动态链接库)
            from win_camera_index.CvCameraIndex import get_camera_index
            for usb_id in usb_ids:
                vid,pid=usb_id["VID"],usb_id["PID"]
                camera_index = get_camera_index('vid_'+vid+'&pid_'+pid) # 'vid_2BDF&pid_0289'
                if camera_index!=-1: # 或者成功
                    if isinstance(cameras[camera_index],str):
                        cameras[camera_index]={
                            'name':cameras[camera_index],
                            'vid':vid,
                            'pid':pid,
                        }
                    else:
                        # 说明有相同 vid pid 的设备
                        cameras[camera_index]=cameras[camera_index]
            # 统一 cameras 的格式为 [ {'name','vid','pid'} ]
            for index,camera in enumerate(cameras):
                if isinstance(camera,str)==True:
                    cameras[index]={
                        'name':cameras[index],
                        'vid':None,
                        'pid':None,
                    }
        elif plat == 'linux':
            # TODO 能否使用 pyusb？
            import serial.tools.list_ports; 
            reslt =  serial.tools.list_ports.comports()
            print(reslt)
            pass
        print(cameras)
        from easydict import EasyDict
        return EasyDict(cameras)

    def can_distinguish_cam_byid(self,camid_maps):
        for camid_map in camid_maps:
            if camid_map['vid']==None or camid_map['pid']==None:
                return False
        return True
    
    def can_distinguish_cam_byname(self,camid_maps):
        names=[]
        for camid_map in camid_maps:
            names.append(camid_map['name'])
        names=list(set(names))
        if len(names)==len(camid_maps):
            return True
        else:
            return False

    def map_capid_real(self,caps,camid_maps):
        # TODO 利用相机名称或pid vid，将虚拟的相机ID转换为真实的相机ID
        # 判断能够通过 id 和 name 区分 cameras
        byid=self.can_distinguish_cam_byid(camid_maps)
        byname=self.can_distinguish_cam_byname(camid_maps)
        if byid==True:
            print("=> you can distinguish cameras by vid&pid, rewrite def map_capid_real in Class MyThreads ..")
        elif byname==True:
            print("=> you can distinguish cameras by name, rewrite def map_capid_real in Class MyThreads ..")
        else:
            print("warning: can't distinguish cameras, virtual index will be used!")
        return caps

    def get_caps(self,resolution):
        camid_maps = self.get_caps_name_id()
        width,height = resolution
        caps = []
        for i in range(self.cam_num):
            cap = self.get_real_cap(i)
            # 设置 视频流 传输格式为MJPG，Opencv默认读取的是YUY2
            # 在Python opencv3版本下，设置摄像头读取视频编码格式需要放在设置其他参数之前
            # 下面这两条语句缺一不可，原因不明 https://blog.csdn.net/qq_46037812/article/details/128668029
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G')) # 通过MJPG格式传输，减轻带宽压力
            # 设置帧率
            cap.set(cv2.CAP_PROP_FPS, 30)
            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            caps.append(cap)
        caps=self.map_capid_real(caps=caps,camid_maps=camid_maps)
        return caps
    
    def get_frames_recs(self,):
        frames=[None]*self.cam_num
        recs=[None]*self.cam_num
        responses=[None]*self.cam_num
        threads=[]
        for i in range(self.cam_num):
            t=MyThread(self.caps[i],self.my_board_detector,self.config)
            threads.append(t)
        for i in range(self.cam_num):
            threads[i].start()
        for i in range(self.cam_num):
            threads[i].join()
        for i in range(self.cam_num):
            frames[i],recs[i],responses[i]=threads[i].get_result()
        return frames,recs,responses
    
    def __del__(self):
        for cap in self.caps:
            cap.release()

class MyCollect():
    def __init__(self,config) :
        self.config=config
        self.img_num=0
        self.my_threads=MyThreads(config)
        self.calibration_img_path=self.config.img_path+"images/"
        self.calibration_all_img_path=self.config.img_path+"all_cams/"
        self.calibration_available_num=[0 for i in range(self.config.cam_num)]
        
    def init_img_dir(self):
        print("=> init calibration image path ..")
        if os.path.exists(self.calibration_img_path):
            shutil.rmtree(self.calibration_img_path)
        if not os.path.exists(self.calibration_img_path):
            os.makedirs(self.calibration_img_path)
        if os.path.exists(self.calibration_all_img_path):
            shutil.rmtree(self.calibration_all_img_path)
        if not os.path.exists(self.calibration_all_img_path):
            os.makedirs(self.calibration_all_img_path)
    
    def save_calibration_img(self,painted_frame,frames):
        print("=> save frames ..") 
        self.img_num+=1
        if self.img_num>50:
            print("you have collected enough image pairs")
        if not os.path.exists(self.calibration_all_img_path):
            os.makedirs(self.calibration_all_img_path)
        saveFile = self.config.img_path+"all_cams/image%02d.jpg"%(self.img_num)
        cv2.imwrite(saveFile, painted_frame)
        for step,frame in enumerate(frames):
            if not os.path.exists(self.calibration_img_path+"cam%01d"%(step+1)):
                os.makedirs(self.calibration_img_path+"cam%01d"%(step+1))
            saveFile = self.calibration_img_path+"cam%01d/"%(step+1)+"image%02d.jpg"%(self.img_num)
            cv2.imwrite(saveFile, frame)

    def collect(self,):
        if self.config.auto==True:
            self.auto_collect()
        else:
            self.manual_collect()

    def auto_collect(self,):
        FPS=0
        self.init_img_dir()
        start_time = time.time()#开始时间
        while True:
            painted_frame,frames,recs,responses=self.painted_frames()
            c=cv2.waitKey(1)
            if c == ord('q'):  # 如果按下q 就退出
                break
            more_than_one_available=0
            for rec in recs:
                if rec==True:
                    more_than_one_available+=1
            if more_than_one_available>=1:
                more_than_one_available=True
            if more_than_one_available==True:
                for index,rec in enumerate(recs):
                    if rec==True:
                        self.calibration_available_num[index]+=1
                self.save_calibration_img(painted_frame,frames) # 保存图片组，自增self.img_num
            end_time = time.time()
            if end_time-start_time <1.0:
                FPS+=1
            else:
                start_time=end_time
                print("\r",end="")   # 输出位置回到行首
                print("程序帧率为：",FPS,"\t 已经收集的图片组数量：",self.img_num,"\t",end="")
                sys.stdout.flush()   # 动态刷新数字，进度条
                FPS=0
            # 对每个相机都收集到足够的数据，结束 auto collect
            stop_collect=True
            for available_num in self.calibration_available_num:
                if available_num<=self.config.max_available_num:
                    stop_collect=False
            if stop_collect == True:
                print("=> 自动化程序采集相机校准图片，执行完毕 .. ")
                break

    def manual_collect(self,):
        FPS=0
        self.init_img_dir()
        start_time = time.time()#开始时间
        while True:
            painted_frame,frames,recs,responses=self.painted_frames()
            c=cv2.waitKey(1)
            if c == ord('q'):  # 如果按下q 就退出
                break
            elif c == ord('r'): # 如果按下r 就重置
                self.init_img_dir()
                self.img_num=0
            elif c == ord('s'):  # 如果按下s 就保存
                for index,frame in enumerate(frames):
                    if recs[index]==True:
                        self.calibration_available_num[index]+=1
                self.save_calibration_img(painted_frame,frames) # 保存图片组，自增self.img_num
            end_time = time.time()
            if end_time-start_time <1.0:
                FPS+=1
            else:
                start_time=end_time
                print("\r",end="")   # 输出位置回到行首
                print("程序帧率为：",FPS,"\t 已经收集的图片组数量：",self.img_num,"\t",end="")
                sys.stdout.flush()   # 动态刷新数字，进度条
                FPS=0
    
    def painted_frames(self,):
        frames,recs,responses=self.my_threads.get_frames_recs()
        painted_frames=[]
        for index,frame in enumerate(frames):
            if self.calibration_available_num[index]>=self.config.max_available_num:
                paint = (0,0,255)*np.ones((frame.shape[0], frame.shape[1], 1), dtype = "uint8") # 红色
                painted_frame=np.array((0.8*frame+0.2*paint),dtype="uint8")
            elif recs[index]==True:
                paint = (0,255,0)*np.ones((frame.shape[0], frame.shape[1], 1), dtype = "uint8") # 绿色
                painted_frame=np.array((0.8*frame+0.2*paint),dtype="uint8")
            else:
                painted_frame=frame
            cv2.putText(
                img = painted_frame,
                text = "calibration available num%02d"%(self.calibration_available_num[index]),
                org = (15, 40),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1, 
                color = (255, 0, 0), # 蓝色
                thickness = 2,
            )
            cv2.putText(
                img = painted_frame,
                text = "available corner point num%03d"%(responses[index]),
                org = (15, 65),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1, 
                color = (255, 0, 0), # 蓝色
                thickness = 2,
            )
            painted_frames.append(painted_frame)
        painted_frame=self.show_multi_imgs(
            scale=0.7,
            imglist=painted_frames,
            order=self.get_order(len(painted_frames))
        )
        cv2.imshow("frame",painted_frame)
        return painted_frame,frames,recs,responses
    
    def get_order(self,len):
        import math
        if len%int(math.sqrt(len))==0:
            columes=int(math.sqrt(len))
        else:
            columes=int(math.sqrt(len))+1
        if len%columes==0:
            rows=int(len/columes)
        else:
            rows=int(len/columes)+1
        return (rows,columes)
    
    def show_multi_imgs(self,scale, imglist, order=None, border=5, border_color=(255, 255, 0)):
        """
        :param scale: float 原图缩放的尺度
        :param imglist: list 待显示的图像序列
        :param order: list or tuple 显示顺序 行×列
        :param border: int 图像间隔距离
        :param border_color: tuple 间隔区域颜色
        :return: 返回拼接好的numpy数组
        """
        if order is None:
            order = [1, len(imglist)]
        allimgs = imglist.copy()
        ws , hs = [], []
        for i, img in enumerate(allimgs):
            if np.ndim(img) == 2:
                allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
            #allimgs[i] = cv2.resize(img,None, fx=scale, fy=scale)
            ws.append(allimgs[i].shape[1])
            hs.append(allimgs[i].shape[0])
        w = max(ws)
        h = max(hs)
        # 将待显示图片拼接起来
        sub = int(order[0] * order[1] - len(imglist))
        # 判断输入的显示格式与待显示图像数量的大小关系
        if sub > 0:
            for s in range(sub):
                allimgs.append(np.zeros_like(allimgs[0]))
        elif sub < 0:
            allimgs = allimgs[:sub]
        imgblank = np.zeros(((h+border) * order[0], (w+border) * order[1], 3)) + border_color
        imgblank = imgblank.astype(np.uint8)
        for i in range(order[0]):
            for j in range(order[1]):
                imgblank[(i * h + i*border):((i + 1) * h+i*border), (j * w + j*border):((j + 1) * w + j*border), :] = allimgs[i * order[1] + j]
        return imgblank

if __name__=="__main__":
    from config import config
    my_collect=MyCollect(config=config)
    my_collect.collect()