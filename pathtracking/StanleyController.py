import numpy as np

class stanley_controller:

    def __init__(self):
        self.k = 3.0  # 增益
        self.L = 0.256  # [m] Wheel base of vehicle
        self.max_steer = 0.6  # [rad] max steering angle
        self.ksoft = 1.0
        self.delta = 0.45
        self.kdelta = 0.18
        self.offset=0
        self.angleoffset=0.01

    def calc_target_index(self,state, cx, cy):
        
        #计算前轴坐标，此处直接传车位坐标即可
        fx = state.location[0] + self.L * np.cos(state.rotation[2])
        fy = state.location[1] + self.L * np.sin(state.rotation[2])

        # 找最近的点，并计算坐标差dx、dy
        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)-2

        # Project RMS error onto front axle vector
        
        front_axle_vec = [-np.cos(state.rotation[2] + np.pi / 2),
                        -np.sin(state.rotation[2] + np.pi / 2)]
        error_front_axle = -self.offset+np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)
        
        return target_idx, error_front_axle
    

    def angle_mod(self,x, zero_2_2pi=False, degree=False):
        if isinstance(x, float):
            is_float = True
        else:
            is_float = False

        x = np.asarray(x).flatten()
        if degree:
            x = np.deg2rad(x)

        if zero_2_2pi:
            mod_angle = x % (2 * np.pi)
        else:
            mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

        if degree:
            mod_angle = np.rad2deg(mod_angle)

        if is_float:
            return mod_angle.item()
        else:
            return mod_angle


    def stanley_control(self , state, cx, cy, cyaw, ck , last_target_idx, last_delta):
        #寻找匹配点index，计算横向误差
        current_target_idx, error_front_axle = self.calc_target_index(state, cx, cy)

        #如果上次的index大于当前找到的index，把上一次的作为本次的匹配点
        # if last_target_idx >= current_target_idx:
        #     current_target_idx = last_target_idx
        # 航向误差的纠正
        # 此处的angle_mod的作用有待考证
            
        e = cyaw[current_target_idx] - state.rotation[2]

        #角度bug解决
        if abs(e) > 3.15:
            if cyaw[current_target_idx] < 0:
                e = 2*np.pi+cyaw[current_target_idx] - state.rotation[2]

        theta_e = self.angle_mod(e)
        # 横向误差的纠正
        theta_d = np.arctan2(self.k * error_front_axle, self.ksoft + state.car_speed)
        # print('e=',theta_e,'d=',theta_d,'d_e=',error_front_axle)
        # Steering control
        delta = theta_e + theta_d 

        #引入差分项，对抗转向机构的延迟
        #引入曲率变化项，控制车身横摆频率
        final_delta = delta  -self.delta*(last_delta - delta)  -self.kdelta*(state.gyro[2]-state.car_speed*ck[current_target_idx])

        if(delta > self.max_steer):
            delta = self.max_steer
        elif(delta < -self.max_steer):
            delta = -self.max_steer

        return final_delta, current_target_idx 
