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

from .MultiShow import show_multi_imgs

class MyAdjustFocus():
    def __init__(self,myCapture,config):
        self.myCapture=myCapture
        self.config=config
    def getImageVar(self,image):
        """
        返回清晰度分数，分数越高，越清晰
        """
        img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
        return imageVar
    def get_frames_vars(self):
        frames=self.myCapture.get_frames()
        frames_vars=[]
        for frame in frames:
            img_var=self.getImageVar(frame)
            cv2.putText(
                img=frame,
                text="Clarity Score %05d"%img_var,
                org = (15, 40),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1, 
                color = (255, 0, 0), # 蓝色
                thickness = 2,
            )
            frames_vars.append(frame)
        return frames_vars
    def manual_adjust_focus(self):
        cv2.namedWindow("painted_frame", cv2.WINDOW_NORMAL)
        while True:
            frames_vars=self.get_frames_vars()
            total_frame=show_multi_imgs(scale=self.config.imgScale,imglist=frames_vars,order=self.config.imgOrder)
            cv2.imshow("painted_frame",total_frame)
            key=cv2.waitKey(1)
            if key==ord('q'):
                break

        

