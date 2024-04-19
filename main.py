from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.basic_shape import QLabsBasicShape
from qvl.qcar import QLabsQCar
from qvl.environment_outdoors import QLabsEnvironmentOutdoors
from qvl.stop_sign import QLabsStopSign
from qvl.traffic_light import QLabsTrafficLight
from qvl.real_time import QLabsRealTime

from pal.products.qcar import QCar
from pal.products.qcar import QCarGPS
import pal.resources.rtmodels as rtmodels

import sys
import time
import math
import cv2
import os
from threading import Thread
from threading import Lock
from queue import Queue
import time
from time import sleep, ctime
import keyboard
from math import *

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pandas import DataFrame 

from PathPlanning.FrenetOptimalPathPlanning import FrenetPathMethod
from pathtracking.PurePursuit import pure_pursuit
from pathtracking.StanleyController import stanley_controller
from scipy.interpolate import splprep,splev
from Lidar_detect.lidar_test import Lidar_detect
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import matplotlib
from YOLO import detect



class Qcar_State_Cartesian:
    def __init__(self):
        self.location=[0,0,0]
        self.rotation=[0,0,0]
        self.car_speed=0
        self.acc=[0,0,0]
        self.gyro=[0,0,0]

#qcar控制量 分别为 油门、转向角、车灯
qcar0=[0,0,False] 
#全局车速
global_car_speed=0.15
#qcar状态变量
qcar_state_cartesian = Qcar_State_Cartesian()
#信号量
map_sig=False  #地图保存标识
stopsign_stop=False #stopsign 停车标识
trafficlight_stop=False
get_state_stop=False#结束状态标志位
#障碍物列表
#obs=[2.2, 0.8, 0.006]
obs=[20.2, 20.8, 20.006]
obs2=[-2.05, 3.071,0.006]


#键盘回调函数
def callback(sign):
    if sign.event_type == 'down' and sign.name == 'k':
        global map_sig
        map_sig=True
    if sign.event_type == 'down' and sign.name == 'esc':
        global get_state_stop
        get_state_stop=True
    if sign.event_type == 'down' and sign.name == 'up':
        qcar0[0]=global_car_speed
    if sign.event_type == 'down' and sign.name == 'down':
        qcar0[0]=-global_car_speed
    if sign.event_type == 'up' and (sign.name == 'up' or sign.name == 'down'):
        qcar0[0]=0
    if sign.event_type == 'down' and sign.name == 'left':
        qcar0[1]=0.4
    if sign.event_type == 'down' and sign.name == 'right':
        qcar0[1]=-0.4
    if sign.event_type == 'up' and (sign.name == 'left' or sign.name == 'right'):
        qcar0[1]=0



#地图处理函数
def map_process():
    with open("./data/globalmap.txt", 'r') as f:
        lines=f.readlines()
    process_x=[]
    process_y=[]
    for line in lines:
        #print(line)
        x,y=line.strip().split(',')
        process_x.append(float(x))
        process_y.append(float(y))  
    print("load succcessful")
    tx, ty, tyaw, tc, csp = FrenetPathMethod.generate_target_course(process_x,process_y)
    f.close
    return tx, ty, tyaw, tc, csp



#地图展示
def map_show():
    tx, ty, tyaw, tc, csp=map_process()
    plt.scatter(tx,ty,marker='.', color='coral')
    x = [-0.12229859914334305, -0.0643133742519453, -0.002962153139544027, 0.06017384402580545, 0.1279899873171206, 0.19957998176355213, 0.2789523467377846, 0.3656232135337013, 0.4693361563252369, 0.5723358400318953, 0.6859724178825005, 0.7989476065224097, 0.9095677642854219, 1.0240229909340746, 1.1438623190919146, 1.280526686024704, 1.4082805160339353, 1.5453720685229673, 1.6705908406752636, 1.8009874445967338, 1.9136784519810153, 2.008184979499254, 2.080533074011859]
    y = [-1.0628145451023905, -1.0474440418463833, -1.003490472806171, -0.9431252417977816, -0.876514287611952, -0.8106720884901806, -0.7524173876906353, -0.7056426448421043, -0.6742553929454941, -0.6598994968549474, -0.6612854891603108, -0.6778799626977008, -0.7057088524840394, -0.7382674342421898, -0.7734851395311872, -0.8071344077636791, -0.8352427509770975, -0.8464660822234932, -0.8380039955715467, -0.7930563177088074, -0.7128064208620624, -0.5995271756940493, -0.4648666958561655]
    plt.scatter(x,y,marker='.', color='red')
    plt.show()



def mapshow_task(queue,lock):
    tx, ty, tyaw, tc, csp=map_process()
    time.sleep(2)
    while True:
        if not queue.empty():
            path = queue.get_nowait()           
        plt.clf()
        plt.plot(tx,ty,'g-')  
        plt.plot(path.x,path.x,'r^')  
        plt.show()
        time.sleep(0.5)
        plt.close()
        if get_state_stop:
            break



#建图任务（一般不用）
def mapping_task(qcar , lock):
    global map_sig
    while True:
        if map_sig:
            lock.acquire()
            statue,location, rotation, scale = qcar.get_world_transform()
            lock.release()
            with open("./data/globalmap.txt", 'a') as f:
                f.write(str(location[0])+','+str(location[1])+'\n')
            print("save succcessful",location)
            f.close
            map_sig =False
        if get_state_stop:
            break
        time.sleep(0.05)



def save_path(path):
    with open("./data/path.txt", 'a') as f:
        f.write(str(path.x)+'|'+str(path.y)+'\n')
    f.close
   


#车辆状态监控任务
def monitor_temp(car,qcar,lock):
    global get_state_stopk
    print("monitor task start!")

    while True:
        # print("acc=",car.accelerometer,
        # "v=",car.batteryVoltage,
        # "gyro=",car.gyroscope,
        # "current=",car.motorCurrent)
        print(qcar.get_world_transform())
        if get_state_stop:
            break
        time.sleep(0.1)



last_point=0
#寻找在全局路径上的匹配点
def find_proper_point(xc,yc,xlist,ylist,csp):
    global last_point
    d_last = 10000
    proper_index = 0

    for i in range( last_point , len(xlist) ):
        d = sqrt ( (xc - xlist[i])**2 + (yc - ylist[i])**2 )
        #print(i,',',d)
        if d_last < d:
            proper_index = i-1
            last_point = i-1
            break
        d_last = d

    proper_s = proper_index*0.05
    proper_theta = csp.calc_yaw(proper_index*0.05)
    proper_x = xlist[proper_index]
    proper_y = ylist[proper_index]
    proper_kappa  = csp.calc_curvature(proper_index*0.05)
    proper_dkappa = csp.calc_dcurvature(proper_index*0.05)
    #print('xlist=',xlist[proper_index],'ylist=', ylist[proper_index])
    #print(proper_index)
    return proper_s,proper_x,proper_y,proper_theta,proper_kappa,proper_dkappa




#路径规划函数
def path_planning_task(qcar_state,path_queue,lock):
    tx, ty, tyaw, tc, csp = map_process()
    PathMethod = FrenetPathMethod()
    global last_point
    global qcar_state_cartesian
    last_point=0
    print("planning task starting....")
    time.sleep(1.5) #此处的延时是因为如果之间启动任务会导致qcar的初始位置获取错误
    print("planning task start!")
    while True:
        #获取车辆的状态信息
        lock.acquire()
        c_x=qcar_state_cartesian.location[0]
        c_y=qcar_state_cartesian.location[1]
        c_theta = qcar_state_cartesian.rotation[2]
        c_speed = qcar_state_cartesian.car_speed
        c_a = qcar_state_cartesian.acc[0]
        c_aside = qcar_state_cartesian.acc[1]
        c_carkappa = c_aside/(c_speed**2)
        lock.release()

        #寻找匹配点
        proper_s,proper_x,proper_y,proper_theta,proper_kappa,proper_dkappa=find_proper_point(c_x, c_y, tx, ty , csp)
        
        #计算当前qcar从笛卡尔坐标系到frenet坐标系转换后的s和l
        c_s,c_l = PathMethod.cartesian_to_frenet2D(proper_s, proper_x, proper_y, proper_theta, proper_kappa, 
                              c_x, c_y, c_speed, c_theta)

        qcar_state.s0 = c_s[0]
        qcar_state.c_speed = c_s[1]
        qcar_state.c_accel = 0
        qcar_state.c_d = c_l[0]
        qcar_state.c_d_d =  0
        qcar_state.c_d_dd = 0

        global obs
        global obs2
        qcar_state.ob = np.array([
                                  [obs[0],obs[1]],
                                  #[obs2[0], obs2[1]]
        ])

        # 输入当前qcar在frenet坐标系下的 s,s_d,s_dd,以及 d,d_d,d_dd 
        # 尝试更改 只输出x，y序列，不输出 k，yaw  使用pure控制算法
        path = PathMethod.frenet_optimal_planning(
                                        csp, qcar_state.s0 , qcar_state.c_speed, qcar_state.c_accel, 
                                        qcar_state.c_d, qcar_state.c_d_d, qcar_state.c_d_dd, qcar_state.ob)
        
        if path != None:
            save_path(path)
        path_queue.put(path)

        if get_state_stop:
            break
        time.sleep(0.05)



#控制函数
def control_task(pal_car,qvl_car,control,path_queue,state,detect_queue,lock):
    global get_state_stopk
    global map_sig
    global qcar_state_cartesian
    global stopsign_stop
    global trafficlight_stop
    count=0
    target_ind=0
    di=0
    last_valid_path=0
    pathsig=False
    stage = 0
    Stanleycontrol=stanley_controller()
    print("control task start!")
    pal_car.read_write_std(0, 0 ,control[2])
    tx, ty, tyaw, tc, csp = map_process()
    detect_result=[]
    labels=[]
    yolopositions=[]
    beliefs=[]
    time.sleep(1)
    while True:
        lock.acquire()
        statue, location, rotation, scale = qvl_car.get_world_transform()
        lock.release()
        #获取转速到车速
        
        for i in range(3):
            qcar_state_cartesian.location[i]=location[i]
            qcar_state_cartesian.rotation[i]=rotation[i]
            qcar_state_cartesian.acc[i]=pal_car.accelerometer[i]
            qcar_state_cartesian.gyro[i]=pal_car.gyroscope[i]
        qcar_state_cartesian.car_speed=pal_car.motorTach 

        if not path_queue.empty():     
            path = path_queue.get_nowait() 
            if path!=None:
                last_valid_path = path #如果规划失败，需要使用上一次生成的路径
            pathsig=True

        
        if not detect_queue.empty():     
            detect_result = detect_queue.get_nowait()

        if len(detect_result) != 0:
            labels,yolopositions,beliefs=[],[],[]
            for i in range(len(detect_result)):
                labels.append(detect_result[i][0])
                yolopositions.append(detect_result[i][1])
                beliefs.append(detect_result[i][2])
        else:
            labels,yolopositions,beliefs=[],[],[]

        #print("control: label=",labels)

        if pathsig:
            k_total=0
            for k in last_valid_path.c:
                k_total +=  abs(k)
            lock.acquire()
            di,target_ind = Stanleycontrol.stanley_control(qcar_state_cartesian,
                                                               last_valid_path.x,
                                                               last_valid_path.y,
                                                               last_valid_path.yaw,
                                                               last_valid_path.c,
                                                               target_ind,
                                                               di)
            lock.release()
            #控制信号输出
            
        #3.65 - 3.95   trafficlight1
        #6.85  stop1
        #11.20 stop2
        #14.0 - 14.4   trafficlight2
        #15.90 finish
            #5.5
            #3.3 3.9
            #4.2
            #6.2
            #10.35
            #13.5 14.35
            #14.4
            #15.2
            proper_carspeed = 10.0/(20+k_total) # +20是用来抑制k_total变化过大带来的影响
            car_speed = proper_carspeed

            if stage == 0 :
                car_speed = proper_carspeed-0.02
                if state.s0 > 3.3 and state.s0 < 3.9:
                    if len(labels)==0:
                        pass
                    else:
                        for label in labels:
                            if label == 1:
                                trafficlight_stop=True
                elif state.s0 > 4.2:
                    stage+=1
                    
            elif stage == 1:
                if state.s0 > 5.70:
                     stopsign_stop = True
                   
            elif stage == 2:
                if state.s0 > 9.9:
                    stopsign_stop = True
           
            elif stage == 3:
                if state.s0 > 13.5 and state.s0 < 14.35:
                    if len(labels)==0:
                        pass
                    else:
                        for label in labels:
                            if label == 1:
                                trafficlight_stop=True
                elif state.s0 > 14.4:
                    stage+=1

            elif stage == 4:
                if state.s0 > 14.80:
                    car_speed = 0

            if stopsign_stop:
                if count < 30:
                    count=count+1
                    car_speed = 0
                else:
                    count = 0
                    car_speed = proper_carspeed
                    stopsign_stop = False
                    stage+=1

            if trafficlight_stop:
                car_speed=0
                for label in labels:
                    if label == 2:
                        car_speed = proper_carspeed
                        trafficlight_stop = False
                        


            #print(car_speed)
            pal_car.read_write_std(car_speed, di ,control[2])
        
        #pal_car.read_write_std(control[0], control[1] ,control[2]) #使用键盘控制
        #print(state.s0)
        #纵向控制不需要标定
        if get_state_stop:
            break
        time.sleep(0.0005)


def Lidar_Task(qcar,lock):
    print("lidar task starting")
    
    lidardetect = Lidar_detect()
    global get_state_stop
    print("lidar task start!")
    while True:
        cluster_center,cluster_radius = lidardetect.lidar_detect(qcar,lock)
        #print(cluster_center,cluster_radius)
        if get_state_stop:
            break
        time.sleep(0.1)
  

def Detect_Task(qcar,detect_queue,lock):
    global get_state_stop
    yolo_detect = detect.detectapi(weights='YOLO/usr_weight/best.pt')
    while True:
        lock.acquire()
        x, camera_image = qcar.get_image(camera=qcar.CAMERA_RGB)
        lock.release()
        result_txt, names , runtime = yolo_detect.detect(camera_image)
        detect_queue.put(result_txt)
        if get_state_stop:
            break
    return


class path_state:
    def __init__(self):
        self.ob = np.array([
                #[2.45 , 1.6],
                [30.0, 6.0]
                ])
        self.c_speed =0  
        self.c_accel = 0.0  
        self.c_d = 0.2 
        self.c_d_d = 0.0  
        self.c_d_dd = 0.0  
        self.s0 = 0.0  



def main():
    os.system('cls')
    
    #与qlab建立连接
    qlabs = QuanserInteractiveLabs()
    cv2.startWindowThread()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost") #与本地建立连接。
    except:
        print("Unable to connect to QLabs")
        return
    print("Connected")
    global obs
    global obs2
    block0=QLabsBasicShape(qlabs)
    block1=QLabsBasicShape(qlabs)

    # block=block0.spawn_degrees(
    #                       location=obs, 
    #                       rotation=[0, 0, 0], 
    #                       scale=[0.1, 0.1, 0.4], 
    #                       configuration=0, 
    #                       waitForConfirmation=True)
    
    # block2=block1.spawn_degrees(
    #                     location=obs2, 
    #                     rotation=[0, 0, 0], 
    #                     scale=[0.1, 0.1, 0.4], 
    #                     configuration=0, 
    #                     waitForConfirmation=True)

    
    #pal链接官方生成的qcar
    car = QCar(            
            0,  # id=
            0,  # readMode=
            600,# frequency=
            0.4,# pwmLimit= 
            0   # steeringBias=
    )

    #qvl链接官方qcar
    qcar = QLabsQCar(qlabs)
    qcar.actorNumber=0
    x_offset = 0.06
    y_offset = 1.75
    qcar.set_transform_and_request_state(location=[-1.335+ x_offset, -2.5+ y_offset, 0.005], 
                                    rotation=[0, 0, -pi/4], 
                                    enableDynamics = True, 
                                    headlights = False, 
                                    leftTurnSignal = False, 
                                    rightTurnSignal = False, 
                                    brakeSignal = False, 
                                    reverseSignal = False,
                                    waitForConfirmation=True)
    lock = Lock()
    global qcar0
    global get_state_stop

    #定义qcar路径规划所需的状态信息
    qcar_path_state=path_state()
    #使用FIFO队列进行线程之间的通信
    path_queue = Queue(maxsize=10)
    detect_queue = Queue(maxsize=10)

    thread_yolo_detect = Thread(target=Detect_Task,args=(qcar,detect_queue,lock))
    thread_yolo_detect.start()

    thread_control = Thread(target=control_task,args=(car,qcar,qcar0,path_queue,qcar_path_state,detect_queue,lock))
    thread_control.start()

    thread_path_planning = Thread(target=path_planning_task,args=(qcar_path_state,path_queue,lock))
    thread_path_planning.start()


    # thread_lidar = Thread(target=Lidar_Task,args=(qcar,lock))
    # thread_lidar.start()


    keyboard.hook(callback)
    time.sleep(1)


    #必须用以下方式停止，否则会出现严重bug
    wait=input("press enter to stop")   
    #QLabsRealTime().terminate_all_real_time_models()

    print("shutdown")

    get_state_stop=True


if __name__ == '__main__':
    main()