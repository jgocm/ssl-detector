from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import object_localization



def predictLinearRegression(xmin, xmax, ymin, ymax, x_weights, y_weights):
    x = [xmin, xmax, ymin, ymax, 1]@x_weights
    y = [xmin, xmax, ymin, ymax, 1]@y_weights
    return x, y

if __name__=="__main__":

    cwd = os.getcwd()
   
    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/camera_matrix_C922.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/camera_distortion_C922.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    camera_distortion = np.loadtxt(PATH_TO_DISTORTION_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = object_localization.Camera(
                camera_matrix=camera_matrix,
                camera_distortion=camera_distortion,
                camera_initial_position=calibration_position
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # LOAD GROUND TRUTH AND SSL VISION POSITIONS
    ground_truth = pd.read_csv(cwd+f'/experiments/9abr/ball_positions.txt', sep=" ", header=None)
    ground_truth_x = ground_truth.iloc[:,0]
    ground_truth_y = ground_truth.iloc[:,1]
    plt.scatter(ground_truth_x, ground_truth_y, label="Ground Truth")

    ssl_vision = pd.read_csv(cwd+f'/experiments/9abr/ssl_vision.txt', sep=" ", header=None)
    ssl_vision_x = ssl_vision.iloc[:,0]
    ssl_vision_y = ssl_vision.iloc[:,1]
    plt.scatter(ssl_vision_x, ssl_vision_y, label="SSL Vision")

    # REGRESS FIELD MARKERS XY RELATIVE POSITIONS TO CAMERA ON FIELD
    field_points = np.loadtxt(cwd+f"/experiments/12abr/field_pixels.txt")

    cam_x = []
    cam_y = []
    for point in field_points:
        pixel_x, pixel_y = point
        # BACK PROJECT PIXEL POSITION TO CAMERA 3D COORDINATES
        object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
        x, y, z = (position[0] for position in object_position)
        cam_x.append(x)
        cam_y.append(y)
    plt.scatter(cam_x, cam_y, label="Marker")

        # BALL AS POINT LINEAR REGRESSION
    regression_weights = np.loadtxt(cwd+f"/experiments/9abr/regression_weights.txt")
    bboxes = np.loadtxt(cwd+f"/experiments/12abr/bounding_boxes.txt")

    # REGRESS BALL XY RELATIVE POSITIONS TO CAMERA ON FIELD
    field_x = []
    field_y = []
    for bounding_box in bboxes:
        [xmin, xmax, ymin, ymax] = bounding_box
        pixel_x, pixel_y = predictLinearRegression(xmin, xmax, ymin, ymax, regression_weights[0], regression_weights[1])
        # BACK PROJECT BALL POSITION TO CAMERA 3D COORDINATES
        object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
        x, y, z = (position[0] for position in object_position)
        field_x.append(x)
        field_y.append(y)
    plt.scatter(field_x, field_y, label="Regression")
    plt.title('Ground-Aware Ball Localization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    

