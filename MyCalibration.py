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

class MyBoardDetector():
    '''
    实例化(__init__)时，根据config文件创建 Calibration Pattern \n
    被调用(detect)时，输入一个图像，输入表示该图像能够用于 Calibration
    '''
    def __init__(self,config):
        self.config=config
        # save Calibration Board Picture
        self.generate()
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
    
    def generate(self,):
        # init path
        self.calibration_board_path=self.config.img_path+"my_boards"
        if os.path.exists(self.calibration_board_path):
            shutil.rmtree(self.calibration_board_path)
        if not os.path.exists(self.calibration_board_path):
            os.makedirs(self.calibration_board_path)
        # 生成 yaml配置文件 、 png图片
        yaml_path=self.generate_yaml()
        png_path=self.generate_png(yaml_path)
        if png_path==None:
            assert False,"调用multical board命令出错！"
        # 读取、展示 校准板图片
        board_png = cv2.imread(png_path)
        board_png = cv2.resize(board_png, (int(board_png.shape[1]/4), int(board_png.shape[0]/4)))
        if self.config.auto==True:
            start=time.time()
            while True:
                cv2.imshow("frame",board_png)
                c=cv2.waitKey(1)
                if c == ord('q'):  # 如果按下q 就退出
                    break
                if time.time()-start>=5.0: # 超过 5s 自动退出
                    break
        else:
            while(True):
                cv2.imshow("frame",board_png)
                c=cv2.waitKey(1)
                if c == ord('q'):  # 如果按下q 就退出
                    break
        cv2.destroyAllWindows() # 释放 校准板图像 占用的窗口

    def shell_command(self,command):
        """
        执行 shell 命令并实时打印输出
        """
        # import pexpect
        # output = pexpect.run(command)
        # output = os.popen(command)
        # print(output.read())
        from subprocess import Popen, PIPE, STDOUT
        process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line)
        exitcode = process.wait()
        return process, exitcode
    
    def generate_png(self,yaml_path):
        png_path=self.config.img_path + "my_boards/"
        yaml_path=os.path.abspath(yaml_path)
        png_path=os.path.abspath(png_path)
        command = " multical boards " 
        command = command + " --boards " + yaml_path 
        command = command + " --paper_size " + self.config.board_size
        command = command + " --pixels_mm " + self.config.board_pixel_pre_mm
        command = command + " --write " + png_path
        print("=> 执行shell命令：",command)
        # 调用 shell命令 multical board
        try:
            process,exitcode=self.shell_command(command=command)
        except:
            return None
        png_path=png_path+"\charuco.png"
        print("=> 校准板 .png图像 被保存在：",png_path)
        return png_path
        
    def generate_yaml(self,):
        from copy import deepcopy
        boards=dict()
        charuco=deepcopy(dict(self.config.board))
        # 补充一下额外的参数
        charuco["_type_"]="charuco"
        charuco["aruco_dict"]="4X4_1000"
        charuco["min_rows"]=3
        charuco["min_points"]=20
        charuco["size"]=charuco["size"]
        boards["charuco"]=deepcopy(charuco)
        # 保存 校准版 yaml文件
        import yaml
        boards={"boards":deepcopy(boards)}
        print("=> 打印校准板参数:",boards)
        with open(self.config.img_path+"my_boards/my_charuco.yaml", 'w') as file:
            file.write(yaml.dump(boards, allow_unicode=True))
        yaml_path=self.config.img_path+"my_boards/my_charuco.yaml"
        print("=> 校准板 .yaml文件 被保存在:",yaml_path)
        return yaml_path

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
            frame = np.full((self.width, self.height, 3), 255, dtype = np.uint8) # 填充空白，让程序能够继续运行
            print("出现带宽限制！填充图像",frame.shape)
            frame=cv2.resize(frame,dsize=(self.width, self.height))
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
        # 在 初始化每个相机的时候， 和 "vid pid", 涉及操作系统底层，分 windows 和 linux 两种情形
        import platform
        plat = platform.system().lower()
        if plat == 'windows':
            # 使用PyCameralist打印所有相机的 "虚拟编号"和"name" 
            from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices
            cameras = list_video_devices()
            for index,camera in enumerate(cameras):
                cameras[index]=camera[1]
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
            # pip install pyserial
            # pip install pyusb
            cameras=[]
            # use v4l Ubuntu下所以的信息都是以文件形成储存，所以可以通过读取设备文件来区别索引号和pid 、vid
            import glob
            cameras_path = '/sys/class/video4linux/'
            cameras_list = list(glob.glob(cameras_path+'video*'))
            cameras_list.sort()
            for index,camera_path in enumerate(cameras_list):
                if index%2==1:
                    continue
                else: # real device index
                    name_path="/name"
                    id_str_path="/device/modalias"
                    name=open(camera_path+name_path,"r").read()
                    id_str=open(camera_path+id_str_path,"r").read()
                    name=name.split(':')[0]
                    id_str=id_str.split(':')[1][0:10]
                    vid=id_str[1:5]
                    pid=id_str[6:]
                    cameras.append({
                        'name':name,
                        'vid':vid,
                        'pid':pid,
                    })       
        print(cameras)
        return cameras
    def can_distinguish_cam_byid(self,camid_maps):
        ids=[]
        for camid_map in camid_maps:
            if camid_map['vid']==None or camid_map['pid']==None:
                return False
            ids.append(camid_map['vid']+camid_map['pid'])
        ids=list(set(ids))
        if len(ids)==len(camid_maps):
            return True
        else:
            return False
    
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
        byid=self.can_distinguish_cam_byid(camid_maps)
        byname=self.can_distinguish_cam_byname(camid_maps)
        if byid==True:
            print("=> you can distinguish cameras by vid&pid, rewrite def map_capid_real in Class MyThreads ..")
            # 利用相机 name区分 cameras
            return caps
        elif byname==True:
            print("=> you can distinguish cameras by name, rewrite def map_capid_real in Class MyThreads ..")
            # 利用相机 vid pid 区分 cameras
            return caps
        else:
            print("warning: can't distinguish cameras, virtual index will be used!")
            exit()
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
        cv2.destroyAllWindows()  # 释放 collect过程中打开的窗口
        yaml_path=self.config.img_path + "my_boards/my_charuco.yaml"
        imgfolder_path=self.config.img_path + "images/"
        return yaml_path,imgfolder_path

    def auto_collect(self,):
        FPS=0
        self.init_img_dir()
        start_time = time.time()#开始时间
        while True:
            painted_frame,frames,recs,responses=self.painted_frames()
            c=cv2.waitKey(1)
            if c == ord('q'):  # 如果按下q 就退出
                break
            if FPS%10 ==0: # 降低满足调试时，保存图片的速率
                more_than_one_available=0
                for rec in recs:
                    if rec==True:
                        more_than_one_available += 1
                if more_than_one_available>=2: # 必须至少有两个摄像头 “共视”，否则不利于校准的效率
                    more_than_one_available=True
                else:
                    more_than_one_available=False
                if more_than_one_available==True:
                    can_save=False
                    for index,rec in enumerate(recs):
                        if rec==True and self.calibration_available_num[index]<=self.config.max_available_num:
                            can_save=True
                    if can_save==True:
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
        self.start_intrinsic_calibrate()
        self.start_extrinsic_calibrate()
        # 返回标准json文件的path
        return os.path.abspath(self.real_json_path)
        
    def start_extrinsic_calibrate(self,):
        # 加载 all cameras 的 相机内参
        with open(self.json_path, "r", encoding="utf-8") as f:
            cams_param = json.load(f)
        # 加载 cam1 的 相机参数
        cam1_intrinsic=EasyDict(cams_param["cameras"]["cam1"])
        cam1_extrinsic=EasyDict(cams_param["camera_poses"]["cam1"])
        # 计算 cam1的 真实 相机外参
        rmat,tvec=self.cam1_extrinsic_calibrate(cam1_param=cam1_intrinsic)
        print("=> cv2.solvePnP 计算得到 cam1 相机外参\n   ",f"R:{list(rmat)}\n   ",f"T:{list(tvec)}")
        # TODO 计算 相机外参 变换矩阵 
        # transform_R=mat(rmat,cam1.R逆)            transform_T=trev - cam1.T
        # cam_new_R=mat(transform_R,cam_origin_R)   cam_new_T=transform_T + cam_origin_T
        cam1_extrinsic_I=np.linalg.inv( np.asarray(cam1_extrinsic.R, dtype = np.float64) )
        transform_R=np.matmul(rmat,cam1_extrinsic_I) # cam1_extrinsic_I 几乎就是 单位阵
        transform_T= np.squeeze(tvec)-np.asarray(cam1_extrinsic.T, dtype = np.float64)
        print("=> transform_R:",transform_R)
        print("=> transform_T:",transform_T)
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
                cam_name=cam_name+"_to_"+"cam%01d"%(index)
                cam_real_param["R"]=cams_param["camera_poses"][cam_name]["R"]
                cam_real_param["T"]=cams_param["camera_poses"][cam_name]["T"]
            cams_real_param.append(cam_real_param)
        # 计算 all cameras 的 真实 相机外参
        for index in range(self.config.cam_num):
            cam_real_param=cams_real_param[index]
            cam_real_param["R"]=np.matmul(transform_R,np.asarray(cam_real_param["R"], dtype = np.float64)).tolist()
            cam_real_param["T"]=(transform_T + np.asarray(cam_real_param["T"], dtype = np.float64)).tolist()
        # print("=> 所有相机的真实参数:",cams_real_param)
        # 保存 all cameras 的 真实 相机外参
        self.real_json_path=self.config.img_path+'real_calibration.json'
        with open(self.real_json_path,'w',encoding='utf8') as f:
            # ensure_ascii=False才能输入中文，否则是Unicode字符
            # indent=2 JSON数据的缩进，美观
            json.dump(cams_real_param,f,ensure_ascii=False,indent=2)
        
    def cam1_extrinsic_calibrate(self,cam1_param):
        print("=> intrinsic calibration得到 所有相机的内参、畸变和相对外参 ..")
        print("=> 假想 cam01 的 外参 是 单位矩阵 ..")
        print("=> 拍摄一张 cam01 的视角，手动逆时针选取长方形的四个点 ..")

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
        print("=> cam1 的相机内参:",cam1_param)
        # 获得来自 cam1 的一帧图像
        if self.config.recollect==True:
            cap= cv2.VideoCapture(0)
            ret, frame = cap.read() 
            cap.release()
            assert ret,"ERROR: can not get view from cam1"
            img_before=frame
        else:
            img_before = cv2.imread('./img_collect/images/cam1/image01.jpg', cv2.IMREAD_COLOR)
        img_before_copy = np.tile(img_before, 1)
        before_coordinate = []
        cv2.namedWindow('collect img_before coordinate',0)
        cv2.setMouseCallback('collect img_before coordinate', mouse_action01, before_coordinate)
        while True:
            # 按esc 或 采集满4个点  退出鼠标采集行为
            if cv2.waitKey(1) == 27 or len(before_coordinate)==4: 
                cv2.destroyAllWindows()
                break
        print("=> 依次选取的像素点位置：",before_coordinate)
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
        cameraMatrix = np.asarray( cam1_param.K ,dtype = np.float64)
        distCoeffs = np.asarray( cam1_param.dist , dtype = np.float64)
        # pnp算法
        retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
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

if __name__=="__main__":
    """
    MyCalibration.py文件中的所有class，都遵循一个单项的“调用”链条
    => MyCalibrate -> MyCollect -> MyThreads -> MyThread -> MyBoardDetector
    """
    from config import config
    my_calibrate=MyCalibrate(config=config)
    json_path=my_calibrate.start_calibrate()
    print("json文件绝对路径：",json_path)
    print("=> TODO 接入 3DHPE 算法 ..")
