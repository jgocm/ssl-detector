import numpy as np
import math
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

def rotateToLocal(global_x, global_y, global_w):
    local_x = global_x*math.cos(global_w) + global_y*math.sin(global_w)
    local_y = -global_x*math.sin(global_w) + global_y*math.cos(global_w)
    local_w = global_w
    return local_x, local_y, local_w

if __name__ == "__main__":

    cwd = os.getcwd()

    quadrado_nr = 2
    frame_nr = 1

    fnames = []

    odometry = []
    movement = [[0, 0, 0]]

    while True:
        try:
            dir = cwd + f"/data/quadrado{quadrado_nr}/{frame_nr}_*.txt"
            fnames.append(glob(dir))

            odometry.append(list(np.loadtxt(fnames[frame_nr-1][0])))
            
            if frame_nr>1:
                movement.append(list(np.subtract(odometry[-1],last_position)))

            last_position = odometry[-1]
            frame_nr += 1
    
        except:
            print(f"Read odometry from {len(odometry)} files!")
            break
    
    odometry_df = pd.DataFrame(odometry, columns=['X', 'Y', 'Theta'])
    movement_df = pd.DataFrame(movement, columns=['X', 'Y', 'Theta'])
    movement_df['Distance'] = [math.sqrt(a[0]**2 + a[1]**2) for a in movement]


    x = list(odometry_df['X'])
    y = list(odometry_df['Y'])

    print(max(movement_df['Distance']))
    plt.scatter(x, y)
    plt.show()
