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
import platform
from easydict import EasyDict
import matplotlib.pyplot as plt 

from .OnceCapture import MyThread

class MyThreads():
    def __init__(self,config):
        self.config=config
        self.cam_num=config.cam_num
        self.resolution=config.resolution
        # 初始化 所有 相机对象
        self.caps=self.get_caps(self.resolution)

    def get_real_cap(self,cam_index):
        plat = platform.system().lower()
        if plat == 'windows':
            platform_index=cam_index
            cap = cv2.VideoCapture(platform_index,cv2.CAP_DSHOW)  # cv2.CAP_DSHOW 是一组初始化参数
            # cap = cv2.VideoCapture(cam_index) 
        elif plat == 'linux':
            platform_index=cam_index*2
            cap = cv2.VideoCapture(platform_index)
        for _ in range(6): # 尝试启动 6次相机
            if cap.isOpened()==True:
                break
            time.sleep(5) # 等待5秒
            cap.open(platform_index)
            print("---- 尝试再次open相机",cam_index," .. ")
        if cap.isOpened()==True:
            print("相机",cam_index,"成功获取VideoCapture对象 .. ")
        else:
            print("Error! 相机",cam_index,"获取VideoCapture对象失败!请重试!")
            sys.exit()
        return cap
    
    def get_caps(self,resolution):
        width,height = resolution
        caps = []
        for i in range(self.cam_num):
            cap = self.get_real_cap(i)
            # 设置 视频流 传输格式为MJPG，Opencv默认读取的是YUY2
            # 在Python opencv3版本下，设置摄像头读取视频编码格式需要放在设置其他参数之前
            # 下面这两条语句缺一不可，原因不明 https://blog.csdn.net/qq_46037812/article/details/128668029
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            # cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G')) # 通过MJPG格式传输，减轻带宽压力
            # cap.set(6,cv2.VideoWriter.fourcc('m','j','p','g')) # 通过MJPG格式传输，减轻带宽压力
            # 设置帧率
            # cap.set(cv2.CAP_PROP_FPS, 30)
            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.5)
            caps.append(cap)
        # 代码储备：可以不用，未来可能通过 vid和pid区分不同的相机
        # camid_maps = self.get_caps_name_id()
        # caps=self.map_capid_real(caps=caps,camid_maps=camid_maps)
        return caps

    
    def get_frames(self,):
        frames=[None]*self.cam_num
        threads=[]
        for i in range(self.cam_num):
            t=MyThread(self.caps[i],self.config)
            threads.append(t)
        for i in range(self.cam_num):
            threads[i].start()
        for i in range(self.cam_num):
            threads[i].join()
        for i in range(self.cam_num):
            frames[i]=threads[i].get_result()
        return frames
    
    def __del__(self):
        for cap in self.caps:
            cap.release()


    """代码储备：可以不用，未来可能通过 vid和pid区分不同的相机"""
    def get_caps_name_id(self,):
        # 在 初始化每个相机的时候， 和 "vid pid", 涉及操作系统底层，分 windows 和 linux 两种情形
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
            return caps
