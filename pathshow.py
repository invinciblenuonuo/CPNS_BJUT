from PathPlanning.FrenetOptimalPathPlanning import FrenetPathMethod
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm
obs=[2.2, 0.8, 0.006]

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
    return tx, ty

class path:
    def __init__(self):
        x=[]
        y=[]

def list_process(str):
    x=[]
    xstr = str.strip('[]\n')
    x = xstr.split(',')
    pathx = [float(num) for num in x]
    return pathx


def read_path():
    with open("./data/path.txt", 'r') as f:
        lines=f.readlines()
    return lines


def main():
    lines=read_path()
    tx, ty = map_process()
    for line in lines:
        strx,stry=line.split('|')
        pathx = list_process(strx)
        pathy = list_process(stry)
        plt.cla()
        plt.plot(tx, ty)
        plt.plot(obs[0], obs[1], "x")        
        plt.plot(pathx[1:], pathy[1:], "-or")
        plt.plot(pathx[1], pathy[1], "vc")
        plt.grid(True)
        plt.pause(0.1)


if __name__ == '__main__':
    main()



