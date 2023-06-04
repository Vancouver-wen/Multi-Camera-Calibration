#涉及到cv2.imshow的代码不能在vscode的terminal中的运行
import cv2
from cv2 import aruco
import numpy as np
import time
import threading
import datetime
import shutil
import os
 
def show_multi_imgs(scale, imglist, order=None, border=5, border_color=(255, 255, 0)):
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

def detect_board(frame):
    # ChAruco board variables
    CHARUCOBOARD_ROWCOUNT = 16
    CHARUCOBOARD_COLCOUNT = 22
    ARUCO_DICT = aruco.getPredefinedDictionary(dict=cv2.aruco.DICT_4X4_1000)
    # Create constants to be passed into OpenCV and Aruco methods
    CHARUCO_BOARD = aruco.CharucoBoard_create(squaresY=CHARUCOBOARD_COLCOUNT,squaresX=CHARUCOBOARD_ROWCOUNT ,squareLength=0.05,markerLength=0.0375,dictionary=ARUCO_DICT)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=ARUCO_DICT)
    #print("=> using thread .. ")
    if corners is ():
        #print("detect no corners")
        return False  #没检测到charuco board，继续运行会报错
    response,_, _ = aruco.interpolateCornersCharuco(markerCorners=corners,markerIds=ids,image=gray,board=CHARUCO_BOARD)
    if response > 20:
        #print("response number enough, return true")
        return True
    else:
        #print("=> ",response,"not enough response ..")
        return False

def detect_board_easy(frame):
    #棋盘格模板规格
    w = 7
    h = 5
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, (w,h),None)
    if success:
        return True
    else:
        return False

class MyThread(threading.Thread):
    def __init__(self, n,cap):
        super(MyThread, self).__init__()  # 重构run函数必须要写
        self.n = n
        self.cap=cap
        self.frame=None
        self.rec=True
 
    def run(self):
        _,self.frame = self.cap.read() #将摄像头拍到的图像作为frame值
        #self.frame = cv2.flip(self.frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示 only for windows
        self.rec=detect_board(self.frame)
        #print(type(self.frame))
        if self.frame is None:
            print("出现带宽限制！",type(self.frame))
            self.frame = np.full((640, 480, 3), 255, dtype = np.uint8)
        #self.rec=detect_board_easy(self.frame)

        #self.frame=cv2.resize(self.frame,dsize=(640,480))
        #print("在线程",self.n,type(self.frame))
        #self.cap.release()  #常规操作
        
    def get_result(self):
        return self.n,self.frame,self.rec


if __name__=="__main__":
    cam_num = 6
    caps=[]
    for i in range(cam_num):
        cap = cv2.VideoCapture(i*2)
        #cap = cv2.VideoCapture(i,cv2.CAP_DSHOW) #CAP_DSHOW是微软特有的,cv2.release()之后摄像头依然开启，需要指定该参数
        cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        caps.append(cap)

    FPS=0
    img_num=0
    start_time = time.time()#开始时间
    while True:   #进入无限循环
        frames=[None]*cam_num
        recs=[None]*cam_num
        threads=[]
        for i in range(cam_num):
            t=MyThread(i,caps[i])
            threads.append(t)
        for i in range(cam_num):
            threads[i].start()
        for i in range(cam_num):
            threads[i].join()
        for i in range(cam_num):
            index,frame,rec=threads[i].get_result()
            frames[index]=frame
            recs[index]=rec
            #print("在主线",i,type(frame))

        end_time = time.time()#结束时间
        if end_time-start_time <1.0:
            FPS+=1
        else:
            start_time=end_time
            print("程序帧率为：",FPS)
            print("recs:",recs)
            print("resolution: ",frames[0].shape)
            FPS=0
        
        frame = show_multi_imgs(0.3, frames, (2, 3))
        cv2.imshow('video',frame) #将frame的值显示出来 有两个参数 前一个是窗口名字，后面是值
        c = cv2.waitKey(1) #判断退出的条件 当按下'Q'键的时候呢，就退出

        # when all cameras can see Charuco Board , save automatically
        auto_save=True
        for i in range(cam_num):
            if recs[i] is False:
                auto_save=False
        if auto_save:  
            print("successfully find a position making Charuco available from all camera")
            img_num+=1
            if img_num>50:
                print("you have collected enough image pairs")
            for i in range(cam_num+1):
                if i==0:
                    saveFile = "./images/cam%01d/image%02d.jpg"%(i,img_num)  # saveFile = "G:\python picture\测试图02.jpg" 带有中文的保存文件路径
                    cv2.imwrite(saveFile, frame)  # imwrite 不支持中文路径和文件名，读取失败，但不会报错!
                else:
                    saveFile = "./images/cam%01d/image%02d.jpg"%(i,img_num)  # saveFile = "G:\python picture\测试图02.jpg" 带有中文的保存文件路径
                    cv2.imwrite(saveFile, frames[i-1])  # imwrite 不支持中文路径和文件名，读取失败，但不会报错!
                #img_write = cv2.imencode(".jpg", frame)[1].tofile(saveFile)
        
        if c == ord('r'):  # 如果按下r 就重置
            img_num=0
            for i in range(cam_num+1):
                path="./images/cam%01d"%i
                shutil.rmtree(path)
                os.mkdir(path)
        if c == ord('q'):  # 如果按下q 就退出
            break

    cv2.destroyAllWindows()