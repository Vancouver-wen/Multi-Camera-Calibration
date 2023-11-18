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

class MyFPSCounter():
    def __init__(self,):
        self.FPS=0
        self.start=time.time()
        self.end=time.time()
    def countFPS(self,):
        self.end=time.time()
        if self.end - self.start >= 1:
            print("=> 当前帧率为：",self.FPS)
            self.FPS=0
            self.start=self.end
        else:
            self.FPS+=1