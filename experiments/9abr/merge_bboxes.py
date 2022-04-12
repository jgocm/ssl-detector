import numpy as np
import os

cwd = os.getcwd()

data_size = 32
bboxes = []

for i in range(1,data_size-1):
    if i>4:     # SKIP DATA Nr. 5
        i = i+1
    bbox = np.loadtxt(cwd+f"/experiments/9abr/{i}.txt")
    if i==28: bbox = np.loadtxt(cwd+f"/experiments/9abr/{data_size}.txt")   # REPLACE DATA Nr. 27 WITH 32
    print(i)

    bboxes.append(bbox)

np.savetxt(cwd+f"/experiments/9abr/bounding_boxes.txt", bboxes)

