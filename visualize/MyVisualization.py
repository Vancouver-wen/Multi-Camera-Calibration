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


class Myplot3d():
    def __init__(self,scale=1):
        import matplotlib.pyplot as plt 
        from mpl_toolkits import mplot3d 
        self.fig = plt.figure(figsize = (10, 7)) 
        self.ax=mplot3d.Axes3D(self.fig) 
        self.ax.set_xlim3d(-500*scale,500*scale)
        self.ax.set_ylim3d(-500*scale,500*scale)
        self.ax.set_zlim3d(0,1000*scale)
        self.ax.set_xlabel("x axis")
        self.ax.set_ylabel("y axis")
        self.ax.set_zlabel("x zxis")
        print("=> 使用 黄色 标记 世界坐标系原点 ..")
        origin_point=(float(0), float(0), float(0))
        self.scatter(origin_point, color="yellow") 
        print("=> 绘制 世界坐标系 X轴Y轴Z轴，分别为 RGB 颜色 ..")
        x_point=(float(100*scale), float(0), float(0))
        y_point=(float(0), float(100*scale), float(0))
        z_point=(float(0), float(0), float(100*scale))
        self.line(origin_point,x_point,"red")
        self.line(origin_point,y_point,"green")
        self.line(origin_point,z_point,"blue")
        # 测试 scatter3D函数 Z轴朝向
        # self.scatter(z_point,"blue")
        #plt.gca().set_box_aspect((3, 5, 2))  # 当x、y、z轴范围之比为3:5:2时
        # plt.gca().set_box_aspect((4, 4, 3))

    def scatter(self,point3d,color="grey"): 
        self.ax.scatter3D(float(point3d[0]), float(point3d[1]), float(point3d[2]), color = color)  
    def scatters(self,points3d):
        """args:
        points3d is a list of tuples. 
        tuple=(x_coordinate,y_coordinate,z_coordinate)
        """
        for point3d in points3d:
            self.scatter(point3d,"grey")
    def line(self,point3d_1,point3d_2,color="gray"):
        line_x=np.array([float(point3d_1[0]),float(point3d_2[0])])
        line_y=np.array([float(point3d_1[1]),float(point3d_2[1])])
        line_z=np.array([float(point3d_1[2]),float(point3d_2[2])])
        self.ax.plot3D(line_x,line_y,line_z,color)
    def lines(self,pointpairs,color="grey"):
        """args:
        pointpairs is a list of tuples. 
        tuples=(point3d_1,point3d_2). point3d is a tuple.
        point3d=(x_coordinate,y_coordinate,z_coordinate)
        """
        for point3d_1,point3d_2 in pointpairs:
            self.line(point3d_1,point3d_2,color)
    def show(self,title="3D scatter plot"):
        plt.title(title)  
        plt.show() 
    # TEST
    def test_scatters(self,):
        z = np.random.randint(80, size =(55))  
        x = np.random.randint(60, size =(55))  
        y = np.random.randint(64, size =(55)) 
        points3d=[]
        for i in range(55):
            points3d.append( (x[i],y[i],z[i]) )
        self.scatters(points3d)
    def test_lines(self,):
        pointpair=[]
        pointpair.append( ((0,0,0),(100,100,100)) )
        self.lines(pointpair)
    def test_scatter(self,):
        point3d=[100,100,100]
        self.scatter(point3d)

def test_Myplot3d():
    myplot3d=Myplot3d()
    myplot3d.test_scatter()
    myplot3d.test_lines()
    myplot3d.show()

class Myshow(Myplot3d):
    def __init__(self,scale=1):
        self.scale=scale
        super().__init__(scale=scale)
    
    def adapter(self,points,map="+x-z+y"):
        """adapter只能在紧挨着 画图代码 之上使用，不能过早调用 \n
        完成坐标系形式转换 \n
        matplotlib默认的坐标系是 (x指向右侧，y指向远离方向，z指向上方) \n
        但 很多时候世界坐标系采用“笛卡尔右手系”或者其他形式 \n
        args: \n
        接受 point=[( , , )] / （ ， ， ）两种格式 和  map="+x-z+y" 两个参数 \n
        map形容点是如何排布的： +x方向指向右侧，-z方向指向远离，+y方向指向上方 \n
        返回值： \n
        point=[(x,y,z)] / （x，y，z）这个点符合matplotlib的需求
        """
        def one_point_adapter(point,map):
            new_point=[None,None,None]
            # 确定 x 的位置
            index=map.find('x')
            if '+' in map[index-1] :
                new_point[0]=point[int(index/2)]
            else:
                new_point[0]=-point[int(index/2)]
            # 确定 y 的位置
            index=map.find('y')
            if '+' in map[index-1] :
                new_point[1]=point[int(index/2)]
            else:
                new_point[1]=-point[int(index/2)]
            # 确定 z 的位置
            index=map.find('z')
            if '+' in map[index-1] :
                new_point[2]=point[int(index/2)]
            else:
                new_point[2]=-point[int(index/2)]
            # list to tuple
            new_point=(new_point[0],new_point[1],new_point[2])
            def not_exist_None(point):
                for i in range(3):
                    if point[i]==None:
                        return False
                return True
            assert not_exist_None(new_point),"请检查 point与map格式"
            return new_point

        if isinstance(points,tuple):
            points=one_point_adapter(points,map)
        elif isinstance(points,list):
            for index in range(len(points)):
                points[index]=one_point_adapter(points[index],map)
        else:
            raise Exception("=> 请检查 points的格式！")
        return points
    def test_adapter(self,):
        """正确的输出结果: \n
        (1, 2, 3)
        [(1, 2, 3), (-1, -2, -3)]
        """
        points=(1,-3,2)
        print(self.adapter(points))
        points=[(1,-3,2),(-1,3,-2)]
        print(self.adapter(points))

    def draw_cameras_position(self,calibrations,map="+x-z+y"):
        hd_cameras=[]
        for calibration in calibrations:
            position=calibration["T"]
            rotation=calibration["R"]
            position=np.array(position)
            rotation=np.array(rotation)
            # 计算原理：
            # 相机坐标系[X,Y,Z] = matmul(R,世界坐标系[X,Y,Z]) + T  等式变换
            # 世界坐标系[X,Y,Z] = matmul(R.-1,相机坐标系[X,Y,Z]-T)  令相机坐标系[X,Y,Z]=[0,0,0]
            # 世界坐标系[X,Y,Z] = matmul(R.-1,-T)
            rotation_inv=np.linalg.inv(rotation)
            position= np.matmul(rotation_inv,-position)
            position=position*100  # 单位换算，从 m 变成 cm
            # 调用adapter，适配matplotlib坐标系
            point3d=self.adapter(point3d,map)
            hd_cameras.append(point3d)
        hd_cameras=[hd_cameras[0]]
        print(hd_cameras)
        self.scatters(hd_cameras)

    def draw_cameras_pose(self,calibrations,map="+x-z+y"):
        """args:
        calibrations的数据结构为: [{"K","R","T","dist","resolution"}] \n
        T 的单位为 m、米
        """
        for calibration in calibrations:
            position=calibration["T"]
            rotation=calibration["R"]
            position=np.array(position)
            rotation=np.array(rotation)
            # 计算原理：
            # 相机坐标系[X,Y,Z] = matmul(R,世界坐标系[X,Y,Z]) + T  等式变换
            # 世界坐标系[X,Y,Z] = matmul(R.-1,相机坐标系[X,Y,Z]-T)  令相机坐标系[X,Y,Z]=[0,0,0]
            # 世界坐标系[X,Y,Z] = matmul(R.-1,-T)
            rotation_inv=np.linalg.inv(rotation)
            position= np.matmul(rotation_inv,-position) # 单位 m
            # 相机坐标系坐标轴
            direction_len=0.2*self.scale # 单位 m
            x_point=np.asarray([float(direction_len), float(0), float(0)],dtype = np.float64) # 相机坐标系
            y_point=np.asarray([float(0), float(direction_len), float(0)],dtype = np.float64)
            z_point=np.asarray([float(0), float(0), float(direction_len)],dtype = np.float64) 
            # 世界坐标系[X,Y,Z] = matmul(R.-1,相机坐标系[X,Y,Z]-T) = matmul(R.-1,相机坐标系[X,Y,Z]) + matmul(R.-1,-T)
            x_direction=np.matmul(rotation_inv,x_point)
            y_direction=np.matmul(rotation_inv,y_point)
            z_direction=np.matmul(rotation_inv,z_point)

            def draw_direction(position,direction,color="gray"):
                # 将单位从 m 换成 cm
                position=position*100
                direction=direction*100 
                direction=direction+position
                # 调用adapter，适配matplotlib坐标系
                position=(position[0],position[1],position[2])
                direction=(direction[0],direction[1],direction[2])
                position=self.adapter(position,map)
                direction=self.adapter(direction,map)
                # 绘制相机位置
                # self.scatter(position)
                # 绘制相机坐标系
                self.line(position,direction,color)
            
            draw_direction(position,x_direction,color="red")
            draw_direction(position,y_direction,color="green")
            draw_direction(position,z_direction,color="blue")

def test_Myshow():
    # 读取 test_vis_calibration.json 转化为标准格式
    with open("test_vis_calibration.json",'r',encoding = 'utf-8') as f:
        calibrations = json.load(f)
    # 整理格式
    calibrations=calibrations["cameras"]
    cameras=[]
    for calibration in calibrations:
        if calibration["type"]=="hd":
            camera=dict()
            camera["resolution"]=calibration["resolution"]
            camera["K"]=calibration["K"]
            camera["R"]=calibration["R"]
            camera["T"]=calibration["t"]
            for index,t in enumerate(camera["T"]):
                t[0]=t[0]/100  # 将单位从 cm 转换成 m
                camera["T"][index]=t[0] # squeeze操作，去除多余维度
            camera["dist"]=calibration["distCoef"]
            cameras.append(camera)
    print(len(cameras))
    print(cameras[0])
    myshow=Myshow()
    # myshow.draw_cameras_position(cameras)
    myshow.draw_cameras_pose(cameras)
    myshow.show()



if __name__=="__main__":
    # test_Myplot3d()
    test_Myshow()

