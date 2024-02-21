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

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame 

from PathPlanning.FrenetOptimalPathPlanning import FrenetPathMethod

#qcar控制量
qcar0=[0,0,False,False,False,False,False] 
#全局车速
global_car_speed=0.01
#信号量
map_sig=False  #地图保存标识
car_stop=False #停车标识
get_state_stop=False#结束状态标志位



#键盘回调函数
def callback(sign):
    if sign.event_type == 'down' and sign.name == 'k':
        global map_sig
        map_sig=True
    if sign.event_type == 'down' and sign.name == 'esc':
        global get_state_stop
        get_state_stop=True
    if sign.event_type == 'down' and sign.name == 'p':
        print("press p")
        global car_stop
        car_stop=not car_stop
    if sign.event_type == 'down' and sign.name == 'up':
        qcar0[0]=global_car_speed
        qcar0[5]=False
    if sign.event_type == 'down' and sign.name == 'down':
        qcar0[0]=-global_car_speed
        qcar0[5]=True
    if sign.event_type == 'up' and (sign.name == 'up' or sign.name == 'down'):
        qcar0[0]=0
        qcar0[5]=False
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
def map_show(path):
    tx, ty, tyaw, tc, csp=map_process()
    plt.scatter(tx,ty,marker='.', color='coral')
    plt.scatter(path.x,path.y,marker='.',color='red')
    plt.show()
    


#控制函数
def control_task(pal_car,qvl_car,control,carLocation_queue,carrotation_queue,path_queue,lock):
    global get_state_stopk
    global map_sig
    print("control task start!")
    while True:
        lock.acquire()
        pal_car.read_write_std(control[0],control[1],control[2])
        statue, location, rotation, scale = qvl_car.get_world_transform()
        #把位姿信息放入消息队列
        carLocation_queue.put(location)
        carrotation_queue.put(rotation)
        #获取规划好的路径
        path = path_queue.get()
        #print(path.x)
        lock.release()
        if get_state_stop:
            break
        time.sleep(0.01)


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


#路径规划函数
def path_planning_task(qcar_state,carstate_queue,carrotation_queue,path_queue,lock):
    tx, ty, tyaw, tc, csp = map_process()
    PathMethod = FrenetPathMethod()
    print("planning task start!")
    while True:
        location=carstate_queue.get()
        rotation=carrotation_queue.get()
        # print('location:',location,'rotation:',rotation)

        #输入当前qcar在frenet坐标系下的 s,s_d,s_dd,以及 d,d_d,d_dd 
        path = PathMethod.frenet_optimal_planning(
        csp, qcar_state.s0 , qcar_state.c_speed, qcar_state.c_accel, 
        qcar_state.c_d, qcar_state.c_d_d, qcar_state.c_d_dd, qcar_state.ob)
        path_queue.put(path)
        #print('path=',path)
        if get_state_stop:
            break
        time.sleep(0.1)



class path_state:
    def __init__(self):
        self.ob = np.array([[20.0, 10.0],
                [30.0, 6.0],
                [30.0, 8.0],
                [35.0, 8.0],
                [50.0, 3.0]
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
    lock = Lock()
    global qcar0
    global get_state_stop

    #定义qcar路径规划所需的状态信息
    qcar_path_state=path_state()

    #使用FIFO队列进行线程之间的通信
    path_queue = Queue(maxsize=1000)
    carstate_queue = Queue(maxsize=1000)
    carrotation_queue = Queue(maxsize=1000)

    thread_control = Thread(target=control_task,args=(car,qcar,qcar0,carstate_queue,carrotation_queue,path_queue,lock))
    thread_control.start()

    thread_path_planning = Thread(target=path_planning_task,args=(qcar_path_state,carstate_queue,carrotation_queue,path_queue,lock))
    thread_path_planning.start()

    # thread_monitor=Thread(target=monitor_temp,args=(car,qcar,lock))
    # thread_monitor.start()

    # thread_mapping = Thread(target=mapping_task,args=(qcar,lock))
    # thread_mapping.start()
    time.sleep(1)
    map_show(path_queue.get())
    keyboard.hook(callback)

    # 必须用以下方式停止，否则会出现严重bug
    wait=input("press enter to stop")
    QLabsRealTime().terminate_all_real_time_models()
    print("shutdown")
    get_state_stop=True




if __name__ == '__main__':
    main()