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


import time
import math
import cv2
import os
from threading import Thread
from threading import Lock
import time
from time import sleep, ctime
import keyboard

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame 

import sys

#qcar控制量
qcar0=[0,0,False,False,False,False,False] 
#全局车速
global_car_speed=0.2
#信号量
map_sig=False  #地图保存标识
car_stop=False #停车标识
get_state_stop=False#结束状态标志位





def callback(sign):
    if sign.event_type == 'down' and sign.name == 'k':
        global imgae_sig
        imgae_sig=True
    if sign.event_type == 'down' and sign.name == 'esc':
        global get_state_stop
        get_state_stop=True
    if sign.event_type == 'down' and sign.name == 'p':
        print("press p")
        global car_stop
        car_stop=not car_stop
    if sign.event_type == 'down' and sign.name == 'o':
        print("press o")
        global traffic_stop
        traffic_stop=not traffic_stop
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


def control_task(qcar,control,lock):
    global get_state_stopk
    global map_sig
    print("control task start!")
    while True:
        lock.acquire()
        qcar.read_write_std(control[0],control[1],control[2])
        lock.release()

        # lock.acquire()
        #statue,location,rotation,scale=qcar.get_world_transform()
        # lock.release()
        # if map_sig:
        #     with open("D:\Documentes\postgraduate\qcar\project_trafficlight_stopsign\map_data.txt", 'a') as f:
        #         f.write(str(b[0])+','+str(b[1])+'\n')
        #         print("save succcessful",b)
        #     f.close
        #     map_sig=False

        if get_state_stop:
            break
        time.sleep(0.01)

def monitor_temp(qcar,lock):
    global get_state_stopk
    while True:
        print("acc=",qcar.accelerometer,
        "v=",qcar.batteryVoltage,
        "gyro=",qcar.gyroscope,
        "current=",qcar.motorCurrent)
        if get_state_stop:
            break
        time.sleep(0.1)

    



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

    #链接官方生成的qcar
    # throttle - 0-pwmlimit
    # steering - 0.6 到 0.6 
    # LEDs - a numpy string of 8 values
    car = QCar(            
            0,  # id=
            0,  # readMode=
            600,# frequency=
            0.4,# pwmLimit=
            0   # steeringBias=
    )
    qcar = QLabsQCar(qlabs)
    qcar.actorNumber=0

    lock = Lock()
    global qcar0
    global get_state_stop

    t1 = Thread(target=control_task,args=(car,qcar0,lock))
    t1.start()

    thread_monitor=Thread(target=monitor_temp,args=(car,lock))
    thread_monitor.start()

    time.sleep(1)
    keyboard.hook(callback)

    #必须用以下方式停止，否则会出现严重bug
    wait=input("press enter to stop")
    QLabsRealTime().terminate_all_real_time_models()
    print("shutdown")
    get_state_stop=True




if __name__ == '__main__':
    main()