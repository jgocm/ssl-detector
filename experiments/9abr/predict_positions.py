from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import object_localization

cwd = os.getcwd()

def predictLinearRegression(xmin, xmax, ymin, ymax, x_weights, y_weights):
    x = [xmin, xmax, ymin, ymax, 1]@x_weights
    y = [xmin, xmax, ymin, ymax, 1]@y_weights
    return x, y

if __name__=="__main__":
   
    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/camera_matrix_C922.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/camera_distortion_C922.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    ssl_cam = object_localization.Camera(
                camera_matrix_path=PATH_TO_INTRINSIC_PARAMETERS,
                camera_distortion_path=PATH_TO_DISTORTION_PARAMETERS,
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # BALL AS POINT LINEAR REGRESSION
    regression_weights = np.loadtxt(cwd+f"/experiments/9abr/regression_weights.txt")
    bboxes = np.loadtxt(cwd+f"/experiments/9abr/bounding_boxes.txt")

    # REGRESS BALL XY RELATIVE POSITIONS TO CAMERA ON FIELD
    cam_x = []
    cam_y = []
    for bounding_box in bboxes:
        [xmin, xmax, ymin, ymax] = bounding_box
        pixel_x, pixel_y = predictLinearRegression(xmin, xmax, ymin, ymax, regression_weights[0], regression_weights[1])
        # BACK PROJECT BALL POSITION TO CAMERA 3D COORDINATES
        object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
        x, y, z = (position[0] for position in object_position)
        cam_x.append(x)
        cam_y.append(y)
    plt.scatter(cam_x, cam_y)

    # LOAD GROUND TRUTH AND SSL VISION POSITIONS
    ground_truth = pd.read_csv(cwd+f'/experiments/9abr/ball_positions.txt', sep=" ", header=None)
    x = ground_truth.iloc[:,0].to_numpy()
    y = ground_truth.iloc[:,1].to_numpy()
    #plt.scatter(x, y)

    ssl_vision = pd.read_csv(cwd+f'/experiments/9abr/ssl_vision.txt', sep=" ", header=None)
    x = ssl_vision.iloc[:,0].to_numpy()
    y = ssl_vision.iloc[:,1].to_numpy()
    plt.scatter(x, y)
        
    plt.show()

