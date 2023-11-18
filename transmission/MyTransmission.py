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

from .save.MySave import MySave
from .redis.MyRedis import MyRedis

class MyTransmission():
    def __init__(self,config):
        self.config=config

        self.save_img=False
        self.redis_img=False

        self.sequence_path=config.save_path
        self.total_frame_path=self.sequence_path+"total_frame/"
        self.frames_path=self.sequence_path+"frames/"
        self.frame_path_list=[self.frames_path+"cam%01d/"%(step+1) for step in range(self.config.cam_num)]
        self.initFolder()
        self.mySave=MySave(config,self.total_frame_path,self.frame_path_list)

        self.myRedis=MyRedis(config)
    
    def initFolder(self,):
        print("=> init sequece folder directory ..")
        if os.path.exists(self.sequence_path):
            shutil.rmtree(self.sequence_path) # 递归地删除文件夹以及里面的文件
        if not os.path.exists(self.sequence_path):
            os.makedirs(self.sequence_path)
        if not os.path.exists(self.total_frame_path):
            os.makedirs(self.total_frame_path)
        if not os.path.exists(self.frames_path):
            os.makedirs(self.frames_path)
        for frame_path in self.frame_path_list:
            if not os.path.exists(frame_path):
                os.makedirs(frame_path)
            
    def set_status(self,key):
        # 开发环境为python3.9
        # 如果是python>=3.10 这里应该使用 match case 语句
        if key==ord('s'): # Save
            print("start save 开始保存图像 .. ")
            self.save_img=True
        if key==ord('r'): # Reset
            print("重置 sequece folder directory ..")
            self.initFolder()
            self.mySave.reset_save_index()
        if key==ord('e'): # End
            print("end save 结束保存图像 .. ")
            self.save_img=False

        if key==ord('t'): # Transmit
            print("redis transmission 将图像传输到 redis server 中 .. ")
            self.redis_img=True
        if key==ord('k'): # Kill
            print("结束 redis transmission")
            self.redis_img=False

        if key==ord('p'): # Print
            print("save图片保存程序运行状态: ",self.save_img)
            print("redis图片传输程序运行状态: ",self.redis_img)

    def transmit(self,key,total_frame,frames):
        # 根据opencv窗口的输入，设定 状态位
        self.set_status(key)
        
        # save 保存图像
        if self.save_img==True:
            self.mySave.save(total_frame,frames)

        # redis 传输图像
        if self.redis_img==True:
            pass # TODO redis传输

if __name__=="__main__":
    pass