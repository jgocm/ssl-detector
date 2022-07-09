from object_localization import KeypointRegression, Camera
from entities import Robot
from navigation import TargetPoint
import object_detection
import tensorrt as trt
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

# OBJECT DETECTION MODEL
PATH_TO_MODEL = cwd+"/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt"
PATH_TO_LABELS = cwd+"/models/ssl_labels.txt"
trt_net = object_detection.DetectNet(
            model_path = PATH_TO_MODEL, 
            labels_path = PATH_TO_LABELS, 
            input_width = 300, 
            input_height = 300,
            score_threshold = 0.3,
            draw = False,
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            )
trt_net.loadModel()

# LOAD GOAL IMAGE AND BOUNDING BOX
STAGE = 3
frame_nr = 300
while True:
    img_path = cwd + f"/data/stage{STAGE}_1/frame{frame_nr}.jpg"
    src = cv2.imread(img_path)
    height, width = src.shape[0], src.shape[1]
    
    detections = trt_net.inference(src).detections

    for detection in detections:
        """
        Detection ID's:
        0: background
        1: ball
        2: goal
        3: robot

        Labels are available at: ssl-detector/models/ssl_labels.txt
        """
        class_id, score, left, right, top,bottom = detection               
        if class_id==2:
            xmin, xmax, ymin, ymax = left, right, top, bottom

    # DETECT GOAL LINE
    coef, intercept = corner_regressor.goalLineRegression(
                                        src=src,
                                        left=xmin,
                                        top=ymin,
                                        right=xmax,
                                        bottom=ymax)

    cv2.line(
            src,
            (0, coef*0 + intercept),
            (640, coef*640 + intercept),
            (255, 0, 0),
            1)

    # TEST SELF ORIENTATION FROM GOAL LINE
    angle = ssl_cam.selfOrientationFromGoalLine(coef, intercept)
    print(angle)

    cv2.imshow('img', src)
    frame_nr += 1
    key = 0xFF & cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()