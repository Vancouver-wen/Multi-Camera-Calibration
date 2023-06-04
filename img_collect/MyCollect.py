import cv2
from cv2 import aruco
import numpy as np
import time
import threading
import datetime
import shutil
import os

class MyBoardDetector():
    def __init__(
            self,
            board_path="..\multical\example_boards\my_charuco_16x22.yaml",
            response_threshold=20,
        ):
        self.response_threshold=response_threshold
        import yaml # pip install pyyaml==5.4.1
        with open(board_path, "r") as f:
            self.board = yaml.load(f.read())
        self.board=self.board["boards"]["charuco_16x22"]
        if self.board["_type_"]=='charuco' and self.board["aruco_dict"]=='4X4_1000':
            # ChAruco board variables
            self.CHARUCOBOARD_ROWCOUNT = self.board["size"][0]
            self.CHARUCOBOARD_COLCOUNT = self.board["size"][1]
            self.SQUARE_LENGTH = self.board["square_length"]
            self.MARKER_LENGTH = self.board["marker_length"]
            self.ARUCO_DICT = aruco.getPredefinedDictionary(dict=cv2.aruco.DICT_4X4_1000)
            # Create constants to be passed into OpenCV and Aruco methods
            self.CHARUCO_BOARD = aruco.CharucoBoard_create(
                squaresX=self.CHARUCOBOARD_ROWCOUNT ,
                squaresY=self.CHARUCOBOARD_COLCOUNT,
                squareLength=self.SQUARE_LENGTH,
                markerLength=self.MARKER_LENGTH,
                dictionary=self.ARUCO_DICT,
            )
        else :
            print("=> 只能接受 charuco aruco_dict=4X4_1000 的输入 .. ")
            exit()
    def detect(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=self.ARUCO_DICT)
        if corners is ():
            # print("=> 没检测到charuco board,detect no corners ..")
            return False
        response,_, _ = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=self.CHARUCO_BOARD
        )
        if response > self.response_threshold:
            # print("response number enough, return true")
            return True
        else:
            # print("=> ",response,"not enough response ..")
            return False


class MyThread(threading.Thread):
    def __init__(
            self, 
            n,cap,
            my_board_detector,
            width=640,height=480,
        ):
        super(MyThread, self).__init__() 
        self.n = n # 相机编号
        self.cap=cap
        self.my_board_detector=my_board_detector
        self.width=width
        self.height=height
        self.frame=None
        self.rec=True # 能够成功检测到校准板
    
    def get_real_frame(self):
        _,frame = self.cap.read()
        # 判断 frame 是否为 None
        if frame is None: # print(type(frame))
            print("出现带宽限制！",type(frame))
            frame = np.full((self.width, self.height, 3), 255, dtype = np.uint8) # 填充空白，让程序能够继续运行
        else:
            frame=cv2.resize(frame,dsize=(self.width, self.height))
        # 判断 运行 平台
        import platform
        plat = platform.system().lower()
        if plat == 'windows':
            pass
            frame = cv2.flip(frame, 1) # 在windows系统中，opencv获取的图像是 镜像 的，需要 flip
        elif plat == 'linux':
            pass
        return frame

    def run(self):
        self.frame=self.get_real_frame()
        self.rec=self.my_board_detector.detect(self.frame)
        
    def get_result(self):
        return self.n,self.frame,self.rec
    
class MyThreads():
    def __init__(self,cam_num,resolution=(640,480)):
        self.cam_num=cam_num
        self.caps=self.get_caps(width=resolution[0],height=resolution[1])
        # 初始化 校准板角点检测器
        self.my_board_detector=MyBoardDetector()

    def get_real_cap(self,cam_index):
        import platform
        plat = platform.system().lower()
        if plat == 'windows':
            cap = cv2.VideoCapture(cam_index,cv2.CAP_DSHOW) 
        elif plat == 'linux':
            cap = cv2.VideoCapture(cam_index*2)
        return cap
    def get_caps(self,width,height):
        caps=[]
        for i in range(self.cam_num):
            cap = self.get_real_cap(i)
            cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G')) # 通过MJPG格式传输，减轻带宽压力
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            caps.append(cap)
        return caps
    def get_frames_recs(self,):
        frames=[None]*self.cam_num
        recs=[None]*self.cam_num
        threads=[]
        for i in range(self.cam_num):
            t=MyThread(i,self.caps[i],self.my_board_detector)
            threads.append(t)
        for i in range(self.cam_num):
            threads[i].start()
        for i in range(self.cam_num):
            threads[i].join()
        for i in range(self.cam_num):
            index,frame,rec=threads[i].get_result()
            frames[index]=frame
            recs[index]=rec
        return frames,recs
    
    def __del__(self):
        for cap in self.caps:
            cap.release()

class MyCollect():
    def __init__(self,cam_num=1) :
        self.my_threads=MyThreads(cam_num)
    def collect(self,img_path="./images/"):
        FPS=0
        img_num=0
        start_time = time.time()#开始时间
        while True:
            painted_frame,frames=self.painted_frames()
            c=cv2.waitKey(1)
            if c == ord('q'):  # 如果按下q 就退出
                break
            elif c == ord('r'): # 如果按下r 就重置
                img_num=0
            elif c == ord('s'):  # 如果按下s 就保存
                print("=> save frames ..") # TODO
                img_num+=1
                if img_num>50:
                    print("you have collected enough image pairs")
                if not os.path.exists(img_path+"all_cam"):
                    os.makedirs(img_path+"all_cam")
                saveFile = img_path+"all_cam/image%02d.jpg"%(img_num)
                cv2.imwrite(saveFile, painted_frame)
                for step,frame in enumerate(frames):
                    if not os.path.exists(img_path+"cam%02d"%(step+1)):
                        os.makedirs(img_path+"cam%02d"%(step+1))
                    saveFile = img_path+"cam%02d/image%02d.jpg"%(step+1,img_num)
                    cv2.imwrite(saveFile, frame)
            end_time = time.time()#结束时间
            if end_time-start_time <1.0:
                FPS+=1
            else:
                start_time=end_time
                print("程序帧率为：",FPS)
                FPS=0
    def painted_frames(self,):
        frames,recs=self.my_threads.get_frames_recs()
        painted_frames=[]
        for index,frame in enumerate(frames):
            if recs[index]==True:
                paint = (0,255,0)*np.ones((frame.shape[0], frame.shape[1], 1), dtype = "uint8") # 绿色
            else:
                paint = (0,0,255)*np.ones((frame.shape[0], frame.shape[1], 1), dtype = "uint8") # 红色
            patinted_frame=np.array((0.8*frame+0.2*paint),dtype="uint8")
            painted_frames.append(patinted_frame)

        painted_frame=self.show_multi_imgs(
            scale=0.7,
            imglist=painted_frames,
            order=self.get_order(len(painted_frames))
        )
        cv2.imshow("frame",painted_frame)
        return painted_frame,frames
    def get_order(self,len):
        import math
        if len%int(math.sqrt(len))==0:
            columes=int(math.sqrt(len))
        else:
            columes=int(math.sqrt(len))+1
        if len%columes==0:
            rows=int(len/columes)
        else:
            rows=int(len/columes)+1
        return (rows,columes)
    def show_multi_imgs(self,scale, imglist, order=None, border=5, border_color=(255, 255, 0)):
        """
        :param scale: float 原图缩放的尺度
        :param imglist: list 待显示的图像序列
        :param order: list or tuple 显示顺序 行×列
        :param border: int 图像间隔距离
        :param border_color: tuple 间隔区域颜色
        :return: 返回拼接好的numpy数组
        """
        if order is None:
            order = [1, len(imglist)]
        allimgs = imglist.copy()
        ws , hs = [], []
        for i, img in enumerate(allimgs):
            if np.ndim(img) == 2:
                allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
            #allimgs[i] = cv2.resize(img,None, fx=scale, fy=scale)
            ws.append(allimgs[i].shape[1])
            hs.append(allimgs[i].shape[0])
        w = max(ws)
        h = max(hs)
        # 将待显示图片拼接起来
        sub = int(order[0] * order[1] - len(imglist))
        # 判断输入的显示格式与待显示图像数量的大小关系
        if sub > 0:
            for s in range(sub):
                allimgs.append(np.zeros_like(allimgs[0]))
        elif sub < 0:
            allimgs = allimgs[:sub]
        imgblank = np.zeros(((h+border) * order[0], (w+border) * order[1], 3)) + border_color
        imgblank = imgblank.astype(np.uint8)
        for i in range(order[0]):
            for j in range(order[1]):
                imgblank[(i * h + i*border):((i + 1) * h+i*border), (j * w + j*border):((j + 1) * w + j*border), :] = allimgs[i * order[1] + j]
        return imgblank



if __name__=="__main__":
    my_collect=MyCollect(cam_num=1)
    my_collect.collect()
    """
    开启一个窗口, 按s保存
    """