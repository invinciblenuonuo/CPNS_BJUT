# from PathPlanning.FrenetOptimalPathPlanning import FrenetPathMethod


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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm

class path:
    def __init__(self):
        x=[]
        y=[]

def list_process(str):
    x=[]
    xstr = str.strip('[]')
    x = xstr.split(',')
    pathx = [float(num) for num in x]
    return pathx


def read_path():
    with open("./data/path.txt", 'r') as f:
        lines=f.readlines()
    return lines



def main():
    lines=read_path()

    for line in lines:
        strx,stry=line.strip().split('|')
        pathx = list_process(strx)
        pathy = list_process(stry)
        plt.cla()
        plt.plot(tx, ty)



if __name__ == '__main__':
    main()



