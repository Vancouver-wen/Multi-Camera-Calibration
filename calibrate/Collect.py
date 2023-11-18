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

from .DetectBoard import MyBoardDetector
from .MultiShow import show_multi_imgs

class MyCollect():
    def __init__(self,config,myCapture):
        self.config=config
        self.img_num=0
        # 初始化 校准板角点检测器
        self.my_board_detector=MyBoardDetector(config)
        # 初始化 Capturer
        self.myCapture=myCapture
        # 初始化 校准图片保存位置
        self.calibration_img_path=self.config.img_path+"images/"
        self.calibration_all_img_path=self.config.img_path+"all_cams/"
        # 用于可视化
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
    
    def get_frames_recs(self):
        frames=self.myCapture.get_frames()
        recs=[]
        responses=[]
        for i in range(len(frames)):
            response=self.my_board_detector.detect(frames[i])
            if response>=self.config.response_threshold:
                rec=True
            else:
                rec=False
            responses.append(response)
            recs.append(rec)
        return frames,recs,responses

    def painted_frames(self,):
        frames,recs,responses=self.get_frames_recs()
        painted_frames=[]
        for index,frame in enumerate(frames):
            if self.calibration_available_num[index]>=self.config.max_available_num and recs[index]==True:
                paint = (0,255,255)*np.ones((frame.shape[0], frame.shape[1], 1), dtype = "uint8") # 红色+绿色
                painted_frame=np.array((0.8*frame+0.2*paint),dtype="uint8")
            elif self.calibration_available_num[index]>=self.config.max_available_num:
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
            painted_frame=show_multi_imgs(scale=self.config.imgScale,imglist=painted_frames,order=self.config.imgOrder)
        return painted_frame,painted_frames,frames,recs,responses
    
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
        cv2.namedWindow("painted_frame", cv2.WINDOW_NORMAL)
        while True:
            painted_frame,painted_frames,frames,recs,responses=self.painted_frames()
            cv2.imshow("painted_frame",painted_frame)
            key=cv2.waitKey(1)
            if key==ord('q'):
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
                print("程序帧率为：",FPS,"   已经收集的图片组数量：",self.img_num)
                FPS=0
            # 对每个相机都收集到足够的数据，结束 auto collect
            stop_collect=True
            for available_num in self.calibration_available_num:
                if available_num < self.config.max_available_num:
                    stop_collect=False
            if stop_collect == True:
                print("=> 自动化程序采集相机校准图片，执行完毕 .. ")
                break

    def manual_collect(self,):
        FPS=0
        self.init_img_dir()
        start_time = time.time()#开始时间
        while True:
            painted_frame,painted_frames,frames,recs,responses=self.painted_frames()

            cv2.namedWindow("painted_frame", cv2.WINDOW_NORMAL)
            cv2.imshow("painted_frame",painted_frame)
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
