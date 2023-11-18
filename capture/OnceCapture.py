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
import platform

class MyThread(threading.Thread):
    """
    被 MyThreads 类调用，只负责采集图像
    """
    def __init__(self,cap,config):
        super(MyThread, self).__init__() 
        self.cap=cap
        self.config=config
        self.width,self.height=self.config.resolution
        self.frame=None

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
        plat = platform.system().lower()
        if plat == 'windows':
            frame=frame
        elif plat == 'linux':
            frame = cv2.flip(frame, 1) # 在linux系统中，opencv获取的图像是 镜像 的，需要 flip
        return frame

    def run(self):
        self.frame=self.get_real_frame()

    def get_result(self):
        return self.frame