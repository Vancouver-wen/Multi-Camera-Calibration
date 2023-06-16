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

from visualize.MyVisualization import Myshow
from collect.Calibration import MyCalibrate


if __name__=="__main__":
    from config import config
    my_calibrate=MyCalibrate(config=config)
    json_path=my_calibrate.start_calibrate() # absolute path
    myplot3d=Myshow(scale=1)
    with open(json_path,'r',encoding = 'utf-8') as f:
        real_calibrations = json.load(f)
    map="+x+y+z"
    myplot3d.draw_cameras_pose(real_calibrations,map)
    myplot3d.show()
    print("=> TODO 接入 3DHPE 算法 ..")
