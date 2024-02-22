"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from PathPlanning.QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from PathPlanning.CubicSpline import cubic_spline_planner



# Parameter
# MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
# MAX_ACCEL = 100.0  # maximum acceleration [m/ss]
# MAX_CURVATURE = 10.0  # maximum curvature [1/m]
# MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
# D_ROAD_W = 1.0  # road width sampling length [m]
# DT = 0.02  # time tick [s] 
# MAX_T = 5.0  # max prediction time [m]
# MIN_T = 4.0  # min prediction time [m]
# TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
# D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
# N_S_SAMPLE = 1  # sampling number of target speed
# ROBOT_RADIUS = 2.0  # robot radius [m]

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 100.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 100.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 0.05 # maximum road width [m]
D_ROAD_W = 0.005  # road width sampling length [m]
DT = 0.2  # time tick [s] 
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 3.0 / 3.6  # target speed [m/s]
D_T_S = 0.5 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 0.02  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.5
K_LON = 1.0

show_animation = True

#五次多项式
class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []



class FrenetPathMethod:
    def __init__(self):
        self.state =1

    def calc_frenet_paths(self,c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
        frenet_paths = []

        for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

            # 横向运动规划
            for Ti in np.arange(MIN_T, MAX_T, DT): 
                fp = FrenetPath() #初始化一个路径类
                #计算五次多项式  起点：xs, vxs, axs 终点： xe, vxe, axe, time
                lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # 纵向运动规划，速度保持
                for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                    TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                    tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                    tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                    tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths


    def calc_global_paths(self,fplist, csp):
        for fp in fplist:

            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # # calc yaw and ds
            # for i in range(len(fp.x) - 1):
            #     dx = fp.x[i + 1] - fp.x[i]
            #     dy = fp.y[i + 1] - fp.y[i]
            #     fp.yaw.append(math.atan2(dy, dx))
            #     fp.ds.append(math.hypot(dx, dy))

            # fp.yaw.append(fp.yaw[-1])
            # fp.ds.append(fp.ds[-1])

            # # calc curvature
            # for i in range(len(fp.yaw) - 1):
            #     fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist


    def check_collision(self,fp, ob):
        for i in range(len(ob[:, 0])):
            d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
                for (ix, iy) in zip(fp.x, fp.y)]

            collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

            if collision:
                return False

        return True


    def check_paths(self,fplist, ob):
        ok_ind = []
        for i, _ in enumerate(fplist):
            #print('v=',fplist[i].s_d,'acc=',fplist[i].s_dd,'CURVATURE=',fplist[i].c)
            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                continue
            elif any([abs(a) > MAX_ACCEL for a in
                    fplist[i].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in
                    fplist[i].c]):  # Max curvature check
                continue
            # elif not self.check_collision(fplist[i], ob):
            #     continue
            ok_ind.append(i)

        return [fplist[i] for i in ok_ind]


    def frenet_optimal_planning(self,csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
        #计算frenet坐标系下的路径
        fplist = self.calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
        fplist = self.calc_global_paths(fplist, csp)
        fplist = self.check_paths(fplist, ob)

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        return best_path


    def generate_target_course(x, y):
        #用四次样条曲线插值全局路径
        #csp 是一个映射
        csp = cubic_spline_planner.CubicSpline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.05)
        
        rx, ry, ryaw, rk = [], [], [], []

        #得到插值后的x，y，heading，kappa
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))
        
        return rx, ry, ryaw, rk, csp


