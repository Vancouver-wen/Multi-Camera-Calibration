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
from joblib import Parallel, delayed

class MySave():
    def __init__(self,config,total_frame_path,frame_path_list):
        self.config=config
        self.total_frame_path=total_frame_path
        self.frame_path_list=frame_path_list
        self.frame_index=0
    def reset_save_index(self,):
        self.frame_index=0

    def save(self,total_frame,frames):
        self.frame_index+=1
        # save total_frame
        self.frame_save(
            frame=total_frame,
            file_name=self.total_frame_path+"image%02d"%(self.frame_index)
        )
        # save frames
        Parallel(n_jobs=self.config.cam_num,backend="threading")(
            delayed(self.frame_save)(
                frame=frame,
                file_name=file_path+"image%02d"%(self.frame_index)
            ) 
            for frame,file_path in list(zip(frames,self.frame_path_list))
        )
    
    def frame_save(self,frame,file_name):
        # 处理 Image Format
        if self.config.img_numpy==True:
            file_name=file_name+".npy"
            np.save(file_name,frame)
        if self.config.img_jpeg==True:
            file_name=file_name+".jpg"
            cv2.imwrite(file_name, frame)


if __name__=="__main__":
    pass