import numpy as np
import os

cwd = os.getcwd()

data_size = 30
bboxes = []

for i in range(1,data_size+1):
    bbox = np.loadtxt(cwd+f"/experiments/12abr/{i}.txt")
    print(i)

    bboxes.append(bbox)

np.savetxt(cwd+f"/experiments/12abr/bounding_boxes.txt", bboxes)

