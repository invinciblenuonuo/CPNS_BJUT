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

class Qcar_State_Cartesian:
    def __init__(self):
        self.location=[0,0,0]
        self.rotation=[0,0,0]
        self.car_speed=0
        self.acc=[0,0,0]

#qcar控制量 分别为 油门、转向角、车灯
qcar0=[0,0,False] 
#全局车速
global_car_speed=0.05
#qcar状态变量
qcar_state_cartesian = Qcar_State_Cartesian()
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



#笛卡尔坐标系转frenet坐标系
# rs,rx,ry,rtheta,rkappa 为参考点的参数
# x,y,v,theta            为当前车的参数
# 返回参数 s[0]:s  s[1]:s'  
def cartesian_to_frenet2D(rs, rx, ry, rtheta, rkappa, x, y, v, theta):
    s_condition = np.zeros(2)
    d_condition = np.zeros(2)
    
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    delta_theta = theta - rtheta
    tan_delta_theta = tan(delta_theta)
    cos_delta_theta = cos(delta_theta)
    
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
    
    
    s_condition[0] = rs 
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d

    return s_condition, d_condition 



def cartesian_to_frenet3D(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
    s_condition = np.zeros(3)
    d_condition = np.zeros(3)
    
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    delta_theta = theta - rtheta
    tan_delta_theta = tan(delta_theta)
    cos_delta_theta = cos(delta_theta)
    
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
    
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    
    d_condition[2] = (-kappa_r_d_prime * tan_delta_theta + 
      one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta *
          (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))
    
    s_condition[0] = rs
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d
    
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_condition[2] = ((a * cos_delta_theta -
                       s_condition[1] * s_condition[1] *
                       (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) /
                          one_minus_kappa_r_d)
    return s_condition, d_condition


#路径规划函数
def path_planning_task(qcar_state,path_queue,lock):
    tx, ty, tyaw, tc, csp = map_process()
    PathMethod = FrenetPathMethod()
    global last_point
    global qcar_state_cartesian
    last_point=0
    print("planning task starting....")
    time.sleep(1) #此处的延时是因为如果之间启动任务会导致qcar的初始位置获取错误
    print("planning task start!")
    while True:
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
        c_s,c_l = cartesian_to_frenet3D(proper_s, proper_x, proper_y, proper_theta, proper_kappa, proper_dkappa, 
                              c_x, c_y, c_speed, c_a , c_theta , c_carkappa )
        print(c_l[0])
        #print('s=',c_s,'l=',c_l)

        # qcar_state.s0 = c_s[0]
        # qcar_state.c_speed = c_s[1]
        # qcar_state.c_accel = c_s[2]

        # qcar_state.c_d = c_l[0]
        # qcar_state.c_d_d =  0
        # qcar_state.c_d_dd = 0
        #求导数和二阶导的算法存在问题

        qcar_state.ob = np.array([
                                  [2.2 , 1.8],
        [30.0, 6.0]
        ])

    
        # 输入当前qcar在frenet坐标系下的 s,s_d,s_dd,以及 d,d_d,d_dd 
        # 尝试更改 只输出x，y序列，不输出 k，yaw  使用pure控制算法
        path = PathMethod.frenet_optimal_planning(
        csp, qcar_state.s0 , qcar_state.c_speed, qcar_state.c_accel, 
        qcar_state.c_d, qcar_state.c_d_d, qcar_state.c_d_dd, qcar_state.ob)

        save_path(path)
        path_queue.put(path)

        qcar_state.s0 = path.s[1]
        qcar_state.c_d = path.d[1]
        qcar_state.c_d_d = path.d_d[1]
        qcar_state.c_d_dd = path.d_dd[1]
        qcar_state.c_speed = path.s_d[1]
        qcar_state.c_accel = path.s_dd[1]

        # print("rs0:",c_s[0],"s0=",path.s[1])
        # print("rc_speed:",c_s[1],"c_speed=",path.s_d[1])
        # print("rc_accel:",c_s[2],"c_accel=",path.s_dd[1])
        
        # print("rc_d:",c_l[1],"c_d=",path.d[1])
        # print("rc_d_d:",c_l[2],"s0=",path.d_d[1])
        # print("rc_d_dd:",c_l[0],"s0=",path.d_dd[1])

        #print('pathx=',path.x,'pathy=',path.y)
        if get_state_stop:
            break
        time.sleep(0.1)


#控制函数
def control_task(pal_car,qvl_car,control,path_queue,lock):
    global get_state_stopk
    global map_sig
    global qcar_state_cartesian
    count=0
    print("control task start!")
    while True:
        #控制信号输出
        pal_car.read_write_std(control[0],control[1],control[2])
        #获取转速到车速
        statue, location, rotation, scale = qvl_car.get_world_transform()
        lock.acquire()
        for i in range(3):
            qcar_state_cartesian.location[i]=location[i]
            qcar_state_cartesian.rotation[i]=rotation[i]
            qcar_state_cartesian.acc[i]=pal_car.accelerometer[i]
        qcar_state_cartesian.car_speed=pal_car.motorTach 
        lock.release()    
        #print(qcar_state_cartesian.rotation[2])
        #从队列中获取规划好的路径
        if not path_queue.empty():
            path = path_queue.get_nowait()           
        #print(path.x)        
        if get_state_stop:
            break
        time.sleep(0.01)


class path_state:
    def __init__(self):
        self.ob = np.array([
                [2.3 , 1.8],
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
    path_queue = Queue(maxsize=10)

    thread_control = Thread(target=control_task,args=(car,qcar,qcar0,path_queue,lock))
    thread_control.start()

    thread_path_planning = Thread(target=path_planning_task,args=(qcar_path_state,path_queue,lock))
    thread_path_planning.start()
   
    # thread_monitor=Thread(target=monitor_temp,args=(car,qcar,lock))
    # thread_monitor.start()

    # thread_mapping = Thread(target=mapping_task,args=(qcar,lock))
    # thread_mapping.start()
    #map_show(path_queue.get())
    keyboard.hook(callback)
    time.sleep(1)
    #map_show()
    # 必须用以下方式停止，否则会出现严重bug
    wait=input("press enter to stop")
    QLabsRealTime().terminate_all_real_time_models()
    print("shutdown")
    get_state_stop=True




if __name__ == '__main__':
    main()