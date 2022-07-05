from object_localization import KeypointRegression, Camera
from entities import Robot
from navigation import TargetPoint
import os
import cv2
import numpy as np
cwd = os.getcwd()
corner_regressor = KeypointRegression()

# CAMERA PARAMETERS SETUP
PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
ssl_cam = Camera(
            camera_matrix=camera_matrix,
            camera_initial_position=calibration_position
            )
points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)
# ROBOT SETUP
ROBOT_ID = 0
ROBOT_HEIGHT = 155
ROBOT_DIAMETER = 180
CAMERA_TO_CENTER_OFFSET = 90
INITIAL_POSE = 0, 0, 0
ssl_robot = Robot(                
            id = ROBOT_ID,
            height = ROBOT_HEIGHT,
            diameter = ROBOT_DIAMETER,
            camera_offset = CAMERA_TO_CENTER_OFFSET,
            initial_pose = INITIAL_POSE
            )
while True:
    # LOAD GOAL IMAGE AND BOUNDING BOX
    STAGE = 3
    frame_nr = 108
    img_path = cwd + f"/data/stage{STAGE}_1/frame{frame_nr}.jpg"
    bbox_path = cwd + f"/data/stage{STAGE}/bbox_frame_nr{frame_nr}.txt"
    src = cv2.imread(img_path)
    height, width = src.shape[0], src.shape[1]
    bbox = np.loadtxt(bbox_path)
    xmin, xmax, ymin, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # DETECT GOAL LINE
    goal_line_points = corner_regressor.goalLineDetection(src=src, 
                                                left=xmin, 
                                                top=ymin, 
                                                right=xmax, 
                                                bottom=ymax)

    for point in goal_line_points:
        pixel_x, pixel_y = point
        #src[pixel_y, pixel_x] = (0, 0, 0)

    # GOAL LINE REGRESSION
    _coef, _intercept = corner_regressor.makeLinearRegressionModel(goal_line_points)
    def predict(coef, intercept, x):
        y = coef*x + intercept
        return y 
    y0 = int(predict(_coef, _intercept, 0))
    y1 = int(predict(_coef, _intercept, width))
    cv2.line(src, (0, y0), (width, y1), (0, 0, 0), 1)

    # COMPUTE SIDE
    side = corner_regressor.getGoalSide(_coef)

    # TEST SELF ORIENTATION FROM GOAL LINE
    angle = ssl_cam.selfOrientationFromGoalLine(_coef, _intercept)
    print(angle)

    cv2.imshow('img', src)
    key = 0xFF & cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()