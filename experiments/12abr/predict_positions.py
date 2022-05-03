from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import object_localization
from sklearn import linear_model
import math


def predictLinearRegression(xmin, xmax, ymin, ymax, x_weights, y_weights):
    x = [xmin, xmax, ymin, ymax, 1]@x_weights
    y = [xmin, xmax, ymin, ymax, 1]@y_weights
    return x, y

def fitDistortionWeights(cam, points3d, points2d):
    y = points3d
    X = []
    for pixel in points2d:
        pixel_x, pixel_y = pixel
        position = cam.pixelToCameraCoordinates(pixel_x, pixel_y, 0)
        p_x, p_y, p_z = (p[0] for p in position)
        X.append([p_x, p_y])
    
    model = linear_model.LinearRegression().fit(X, y)
    coefficients = model.coef_
    intercept = model.intercept_
    weights_x = np.append(coefficients[0],intercept[0])
    weights_y = np.append(coefficients[1],intercept[1])
    return weights_x, weights_y

def predictUndistortion(point_x, point_y, weights_x, weights_y):
    x = [point_x, point_y, 1]@weights_x
    y = [point_x, point_y, 1]@weights_y
    #print (f'{point_x},{point_y} -> {x}, {y}')
    return x, y

def fixHeightAngle(h1, alfa1, h2):
    alfa1 = 180-alfa1
    alfa2 = math.atan(h1*math.tan(alfa1)/h2)
    print(alfa2)
    return alfa2

if __name__=="__main__":

    cwd = os.getcwd()

    # STOP ON POINT 'last_nr'
    init_nr = 0
    last_nr = 30

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    camera_distortion = np.loadtxt(PATH_TO_DISTORTION_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = object_localization.Camera(
                camera_matrix=camera_matrix,
                #camera_distortion=camera_distortion,
                camera_initial_position=calibration_position
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # LOAD GROUND TRUTH AND SSL VISION POSITIONS
    ground_truth = pd.read_csv(cwd+f'/experiments/9abr/ball_positions.txt', sep=" ", header=None)
    ground_truth_x = ground_truth.iloc[init_nr:last_nr,0].to_numpy()
    ground_truth_y = ground_truth.iloc[init_nr:last_nr,1].to_numpy()
    plt.scatter(ground_truth_x, ground_truth_y, label="Ground Truth")

    ssl_vision = pd.read_csv(cwd+f'/experiments/9abr/ssl_vision.txt', sep=" ", header=None)
    ssl_vision_x = ssl_vision.iloc[init_nr:last_nr,0]
    ssl_vision_y = ssl_vision.iloc[init_nr:last_nr,1]
    plt.scatter(ssl_vision_x, ssl_vision_y, label="SSL Vision")

    # TEST FIX CAMERA HEIGHT

    # REGRESS FIELD MARKERS XY RELATIVE POSITIONS TO CAMERA ON FIELD
    field_points = np.loadtxt(cwd+f"/experiments/12abr/field_pixels.txt")

    field_x = []
    field_y = []
    for point in field_points[init_nr:last_nr]:
        pixel_x, pixel_y = point
        # BACK PROJECT PIXEL POSITION TO CAMERA 3D COORDINATES
        object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
        x, y, z = (position[0] for position in object_position)
        field_x.append(x)
        field_y.append(y)
    #plt.scatter(field_x, field_y, label="Marker")

    # BALL AS POINT LINEAR REGRESSION
    regression_weights = np.loadtxt(cwd+f"/experiments/12abr/regression_weights.txt")
    bboxes = np.loadtxt(cwd+f"/experiments/12abr/bounding_boxes.txt")

    # REGRESS BALL XY RELATIVE POSITIONS TO CAMERA ON FIELD
    ball_x = []
    ball_y = []
    theta = []
    for bounding_box in bboxes[init_nr:last_nr]:
        [xmin, xmax, ymin, ymax] = bounding_box
        pixel_x, pixel_y = predictLinearRegression(xmin, xmax, ymin, ymax, regression_weights[0], regression_weights[1])
        # BACK PROJECT BALL POSITION TO CAMERA 3D COORDINATES
        object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
        x, y, z = (position[0] for position in object_position)
        ball_x.append(x)
        ball_y.append(y)
        rad = math.atan(x/y)
        theta.append(math.degrees(rad))
    np.savetxt(cwd+f"/experiments/13abr/theta.txt", theta)
    plt.scatter(ball_x, ball_y, label="On-Board Vision")

    e2 = (ball_x-ground_truth_x)**2 + (ball_y-ground_truth_y)**2
    RMSE = math.sqrt(sum(e2)/len(e2))
    print(f'Ball RMSE:{RMSE}')

    x0, y0 = 0, -500
    d2 = (ground_truth_x-x0)**2 + (ground_truth_y-y0)**2
    norm_e2 = e2/d2
    RMSE = math.sqrt(sum(norm_e2)/len(norm_e2))
    #print(RMSE)

    # LINEAR REGRESSION FOR DISTORTION CORRECTION
    points3d = ssl_cam.fixPoints3d(points3d)

    weights_x, weights_y = fitDistortionWeights(ssl_cam, points3d[:,:-1], points2d)

    ball = np.column_stack((ball_x, ball_y))
    fix_x = []
    fix_y = []
    for point in ball:
        point_x, point_y = point[0], point[1]
        fix_x.append(predictUndistortion(point_x, point_y, weights_x, weights_y)[0])
        fix_y.append(predictUndistortion(point_x, point_y, weights_x, weights_y)[1])

    #plt.scatter(fix_x, fix_y, label="Ball Distortion Fix")
    
    #print("Post distortion correction")
    e2 = (fix_x-ground_truth_x)**2 + (fix_y-ground_truth_y)**2
    RMSE = math.sqrt(sum(e2)/len(e2))
    #print(f'Ball RMSE:{RMSE}')

    field = np.column_stack((field_x, field_y))
    fix_x = []
    fix_y = []
    for point in field:
        point_x, point_y = point[0], point[1]
        fix_x.append(predictUndistortion(point_x, point_y, weights_x, weights_y)[0])
        fix_y.append(predictUndistortion(point_x, point_y, weights_x, weights_y)[1])

    #plt.scatter(fix_x, fix_y, label="Marker Distortion Fix")

    # PLOT ROBOT POSITION
    #plt.scatter(0, 0)

    plt.title('Ground-Aware Ball Localization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()