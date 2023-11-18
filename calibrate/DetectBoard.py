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

class MyBoardDetector():
    '''
    实例化(__init__)时，根据config文件创建 Calibration Pattern \n
    被调用(detect)时，输入一个图像，输出表示该图像能够用于 Calibration
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
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
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