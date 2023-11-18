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

from capture.MultiCapture import MyThreads as MyCapture
from calibrate.MultiShow import show_multi_imgs
from visualize.MyVisualization import Myshow

from visualize.MyFPSCounter import MyFPSCounter
from transmission.MyTransmission import MyTransmission

if __name__=="__main__":
    from config import config
    # 相机采集对象
    myCapture=MyCapture(config)

    # 调焦
    if config.adjustFocus==True:
        from calibrate.AdjustFocus import MyAdjustFocus
        myAdjustFocus=MyAdjustFocus(myCapture,config)
        myAdjustFocus.manual_adjust_focus()

    # 校准
    if config.recollect==True:
        from calibrate.Collect import MyCollect
        print("开始采集校准图片 .. ")
        myCollect=MyCollect(config,myCapture)
        yaml_path,imgfolder_path=myCollect.collect()
        from calibrate.Calibration import MyCalibration
        print("校准图片采集完成，开始校准 .. ")
        myCalibration=MyCalibration(config,yaml_path,imgfolder_path)
        cams_params_json_path=myCalibration.start_calibrate(myCapture)
        pass
        print("校准完成！")
    else:
        # 使用之前的校准文件
        cams_params_json_path=".\img_collect\\real_calibration.json"
    
    # 可视化 相机校准json文件
    myplot3d=Myshow(scale=1)
    with open(cams_params_json_path,'r',encoding = 'utf-8') as f:
        real_calibrations = json.load(f)
    map="+x+y+z"
    myplot3d.draw_cameras_pose(real_calibrations,map)
    myplot3d.show()
    
    # 初始化 FPS Conter
    myFPSCounter=MyFPSCounter()
    # 初始化 传输transmission(save/redis)
    myTransmission=MyTransmission(config)
    # 采集动作捕捉图像
    cv2.namedWindow("total_frame", cv2.WINDOW_NORMAL)
    while True:
        # 打印帧率
        myFPSCounter.countFPS()
        # 获取图像
        frames=myCapture.get_frames()
        # 拼接展示图像
        total_frame=show_multi_imgs(scale=config.imgScale,imglist=frames,order=config.imgOrder)
        # 展示图像
        cv2.imshow("total_frame",total_frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
        # 传输(save/redis)
        myTransmission.transmit(key,total_frame,frames) 
        
    del myCapture

        

