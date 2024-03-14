import math

class pure_pursuit:
    
    def __init__(self) -> None:     
        self.k = 0.1   # 前视比例
        self.Lfc = 0.4  # 前视距离
        self.Kp = 1.0   # 速度比例
        self.dt = 0.1   # 
        self.L = 0.256  # 轴距  
    
    def calc_target_index(self,state, cx, cy):

        '寻找最近的点'
        #遍历所有点，将到所有点到当前车坐标的距离保存
        dx = [state.location[0] - icx for icx in cx]
        dy = [state.location[1] - icy for icy in cy]

        #计算距离
        d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]

        #寻找最近的点的序号
        ind = d.index(min(d))

        L = 0.0

        Lf = self.Lfc + self.k*state.car_speed

        '寻找前视点'

        while Lf > L and (ind + 1) < len(cx): #如果最近的点与当前遍历到的点的距离小于前世距离就继续循环
            dx = cx[ind + 1] - cx[ind]        #两点坐标相对距离
            dy = cy[ind + 1] - cy[ind]  
            L += math.sqrt(pow(dx,2) + pow(dy,2)) #累加距离
            ind += 1 

        return ind


    def pure_pursuit_control(self, state, cx, cy, pind):

        ind = self.calc_target_index(state, cx, cy)

        # if pind >= ind:
        #     ind = pind

        if ind < len(cx):
            tx = cx[ind]
            ty = cy[ind]
        else:
            tx = cx[-1]
            ty = cy[-1]
            ind = len(cx) - 1

        alpha = math.atan2(ty - state.location[1], tx - state.location[0]) - state.rotation[2]

        if state.car_speed < 0:  # back
            alpha = math.pi - alpha

        Lf = self.Lfc + self.k*state.car_speed

        delta = math.atan2(2.0 * self.L * math.sin(alpha) / Lf, 1.0)

        return delta, ind
