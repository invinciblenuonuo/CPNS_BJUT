# region: package imports
import time
import numpy as np
import os

import keyboard
from sklearn.cluster import DBSCAN
# import hdbscan
from scipy.ndimage import gaussian_filter

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import matplotlib

matplotlib.use('Qt5Agg')  # 选择TkAgg作为后端，也可以尝试其他后端
import matplotlib.pyplot as plt

# environment objects

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.flooring import QLabsFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.crosswalk import QLabsCrosswalk
import pal.resources.rtmodels as rtmodels

# endregion

# Try to connect to Qlabs
os.system('cls')
qlabs = QuanserInteractiveLabs()
print("Connecting to QLabs...")
try:
    qlabs.open("localhost")
    print("Connected to QLabs")
except:
    print("Unable to connect to QLabs")
    quit()

# Delete any previous QCar instances and stop any running spawn models
qlabs.destroy_all_spawned_actors()
# QLabsRealTime().terminate_all_real_time_models()

# Set the Workspace Title
hSystem = QLabsSystem(qlabs)
x = hSystem.set_title_string('ACC Self Driving Car Competition', waitForConfirmation=True)

### Flooring

x_offset = 0.13
y_offset = 1.67
hFloor = QLabsFlooring(qlabs)
# hFloor.spawn([0.199, -0.491, 0.005])
hFloor.spawn_degrees([x_offset, y_offset, 0.001], rotation=[0, 0, -90])

### region: Walls
hWall = QLabsWalls(qlabs)
hWall.set_enable_dynamics(False)

for y in range(5):
    hWall.spawn_degrees(location=[-2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001], rotation=[0, 0, 0])

for x in range(5):
    hWall.spawn_degrees(location=[-1.9 + x + x_offset, 3.05 + y_offset, 0.001], rotation=[0, 0, 90])

for y in range(6):
    hWall.spawn_degrees(location=[2.4 + x_offset, (-y * 1.0) + 2.55 + y_offset, 0.001], rotation=[0, 0, 0])

for x in range(5):
    hWall.spawn_degrees(location=[-1.9 + x + x_offset, -3.05 + y_offset, 0.001], rotation=[0, 0, 90])

hWall.spawn_degrees(location=[-2.03 + x_offset, -2.275 + y_offset, 0.001], rotation=[0, 0, 48])
hWall.spawn_degrees(location=[-1.575 + x_offset, -2.7 + y_offset, 0.001], rotation=[0, 0, 48])

# Spawn a QCar at the given initial pose
car2 = QLabsQCar(qlabs)
car2.spawn_id_degrees(actorNumber=0, location=[-1.335 + x_offset, -2.5 + y_offset, 0.005], rotation=[0, 0, -45],
                      scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
basicshape2 = QLabsBasicShape(qlabs)
basicshape2.spawn_id_and_parent_with_relative_transform(actorNumber=102, location=[1.15, 0, 1.8], rotation=[0, 0, 0],
                                                        scale=[.65, .65, .1], configuration=basicshape2.SHAPE_SPHERE,
                                                        parentClassID=car2.ID_QCAR, parentActorNumber=2,
                                                        parentComponent=1, waitForConfirmation=True)
basicshape2.set_material_properties(color=[0.4, 0, 0], roughness=0.4, metallic=True, waitForConfirmation=True)

camera1 = QLabsFreeCamera(qlabs)
camera1.spawn_degrees(location=[-0.426 + x_offset, -5.601 + y_offset, 4.823], rotation=[0, 41, 90])

camera2 = QLabsFreeCamera(qlabs)
camera2.spawn_degrees(location=[-0.4 + x_offset, -4.562 + y_offset, 3.938], rotation=[0, 47, 90])

camera3 = QLabsFreeCamera(qlabs)
camera3.spawn_degrees(location=[-0.36 + x_offset, -3.691 + y_offset, 2.652], rotation=[0, 47, 90])

camera2.possess()

# stop signs
myStopSign = QLabsStopSign(qlabs)
myStopSign.spawn_degrees([2.25 + x_offset, 1.5 + y_offset, 0.05], [0, 0, -90], [0.1, 0.1, 0.1], False)
myStopSign.spawn_degrees([-1.3 + x_offset, 2.9 + y_offset, 0.05], [0, 0, -15], [0.1, 0.1, 0.1], False)

# Spawning crosswalks
myCrossWalk = QLabsCrosswalk(qlabs)
myCrossWalk.spawn_degrees(location=[-2 + x_offset, -1.475 + y_offset, 0.01],
                          rotation=[0, 0, 0], scale=[0.1, 0.1, 0.075],
                          configuration=0)

mySpline = QLabsBasicShape(qlabs)
mySpline.spawn_degrees([2.05 + x_offset, -1.5 + y_offset, 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)
mySpline.spawn_degrees([-2.075 + x_offset, y_offset, 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)

# Start spawn model
QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)

# To terminate the spawn model use the following command:
# QLabsRealTime().terminate_real_time_model(rtmodels.QCAR_STUDIO)

# Spawn a QCar at the given initial pose
car10 = QLabsQCar(qlabs)
car10.spawn_id_degrees(actorNumber=10, location=[0.122, 0.891, 0.005], rotation=[0, 0, 0],
                       scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)

cube0 = QLabsBasicShape(qlabs)
# spawn a second cube using degrees
cube0.spawn_id_degrees(actorNumber=10, location=[1.575, 0.812, 0.1], rotation=[0, 0, 45], scale=[0.15, 0.15, 0.2],
                       configuration=cube0.SHAPE_CUBE, waitForConfirmation=True)

cube1 = QLabsBasicShape(qlabs)
# spawn a second cube using degrees
cube1.spawn_id_degrees(actorNumber=11, location=[0.28, -0.39, 0.1], rotation=[0, 0, 0], scale=[0.15, 0.15, 0.2],
                       configuration=cube1.SHAPE_CUBE, waitForConfirmation=True)

cylinder0 = QLabsBasicShape(qlabs)
# spawn a second cube using degrees
cylinder0.spawn_id_degrees(actorNumber=100, location=[-1.195, 0.837, 0.1], rotation=[0, 0, 45], scale=[0.15, 0.15, 0.2],
                           configuration=cylinder0.SHAPE_CYLINDER, waitForConfirmation=True)

cylinder1 = QLabsBasicShape(qlabs)
# spawn a second cube using degrees
cylinder1.spawn_id_degrees(actorNumber=101, location=[0.048, 2.341, 0.1], rotation=[0, 0, 45], scale=[0.15, 0.15, 0.2],
                           configuration=cylinder1.SHAPE_CYLINDER, waitForConfirmation=True)

lidar_rate = 0.01
# Creating a plot to plot the LIDAR data
lidarPlot = pg.plot(title="LIDAR")
squareSize = 10
lidarPlot.setXRange(-squareSize, squareSize)
lidarPlot.setYRange(-squareSize, squareSize)
lidarData = lidarPlot.plot([], [], pen=None, symbol='o', symbolBrush='r', symbolPen=None, symbolSize=2)

time.sleep(1)

print("Reading from LIDAR... if QLabs crashes or output isn't great, make sure FPS > 100 or fix the crash bug!")

plt.show()
vel = 0
th = 0
# Obtaining and plotting lidar data for 0.2s
# for count in range(20):
while True:
    success, angle, distance = car10.get_lidar(samplePoints=720)

    x = np.sin(angle) * distance
    y = np.cos(angle) * distance

    lidarData.setData(x, y)
    QtWidgets.QApplication.instance().processEvents()

    cloud_data = np.column_stack((x, y))
    smoothed_data = gaussian_filter(cloud_data, sigma=0.1)

    # 创建DBSCAN模型
    dbscan = DBSCAN(eps=0.03, min_samples=5)
    # hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)

    # 拟合模型并获取聚类结果
    labels = dbscan.fit_predict(smoothed_data)

    # 获取唯一的标签
    unique_labels = np.unique(labels)

    # 可视化结果
    for label in unique_labels:
        if label == -1:
            plt.scatter(cloud_data[labels == label, 0], cloud_data[labels == label, 1], color='gray', label='Noise')
        else:
            cluster_points = cloud_data[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_radius = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))

            print(f"Cluster {label} - Center: {cluster_center}, Radius: {cluster_radius}")
            if cluster_radius < 0.25:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
                plt.scatter(cluster_center[0], cluster_center[1], marker='x', label=f'Center {label}')

    # plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('DBSCAN Clustering')
    plt.pause(0.1)  # 等待0.1秒，确保图形有足够的时间显示
    plt.clf()  # 清除当前图形，准备下一次迭代

    if keyboard.is_pressed('w'):
        vel = vel + 0.1
        if th > 0:
            th = th - 0.02
        else:
            th = th + 0.02
    elif keyboard.is_pressed('a'):
        vel = vel
        th = th - 0.02
    elif keyboard.is_pressed('s'):
        vel = vel - 0.1
        if th > 0:
            th = th - 0.02
        else:
            th = th + 0.02
    elif keyboard.is_pressed('d'):
        vel = vel
        th = th + 0.02
    else:
        if vel > 0:
            vel = vel - 0.2
        elif vel < 0:
            vel = vel + 0.2
        if th > 0:
            th = th - 0.05
        elif th < 0:
            th = th + 0.05
    car10.set_velocity_and_request_state(forward=vel, turn=th, headlights=True, leftTurnSignal=False,
                                         rightTurnSignal=True, brakeSignal=False, reverseSignal=False)

    time.sleep(lidar_rate)  # lidar_rate is set at the top of this example
