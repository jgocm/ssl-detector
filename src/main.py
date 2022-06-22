import math
import cv2
import numpy as np
import tensorrt as trt
import time
import socket
import argparse
import sys
import os

# LOCAL IMPORTS
from entities import Robot, Goal, Ball, Frame
import object_detection
import object_localization
import communication_proto
import interface
from navigation import TargetPoint

def rotationSign(x, y):
    w = math.atan2(-y, -x)
    if w>0:
        return -1
    else:
        return 1

def offsetTarget(x, y, offset=1):
    dist = math.sqrt(x**2+y**2) + 0.001
    prop = (dist - offset)/dist
    target_x, target_y = prop*x, prop*y
    return target_x, target_y

def directionVector(x1, y1, x2, y2):
    vy = y2-y1
    vx = x2-x1
    w = math.atan2(-vy, -vx)
    return vx, vy, w

def alignedTarget(vx, vy, bx, by, offset):
    v_norm = math.sqrt(vx**2+vy**2)
    target_x = vx*offset/v_norm + bx
    target_y = vy*offset/v_norm + by
    target_w = math.atan2(target_y, target_x)
    return target_x, target_y, target_w

def main():
    cwd = os.getcwd()

    # START TIME
    start = time.time()

    # DISPLAY TITLE
    WINDOW_NAME = 'Vision Blackout'
    SHOW_DISPLAY = True

    # DISPLAYS POSITIONS AND MARKERS ON SCREEN
    DRAW = True

    # ROBOT SETUP
    ROBOT_ID = 0
    ROBOT_HEIGHT = 155
    ROBOT_DIAMETER = 180
    CAMERA_TO_CENTER_OFFSET = 90
    ssl_robot = Robot(                
                id = ROBOT_ID,
                height = ROBOT_HEIGHT,
                diameter = ROBOT_DIAMETER,
                camera_offset = CAMERA_TO_CENTER_OFFSET
                )
    ssl_robot.charge = True

    # INIT ENTITIES
    ssl_ball = Ball()
    ssl_goal = Goal()
    target = TargetPoint()

    # UDP COMMUNICATION SETUP
    HOST_ADDRES = "199.0.1.2"
    HOST_PORT = 9601
    DEVICE_ADRESS = "199.0.1.1"
    DEVICE_PORT = 9600
    eth_comm = communication_proto.SocketUDP(
        host_address=HOST_ADDRES,
        host_port=HOST_PORT,
        device_address=DEVICE_ADRESS,
        device_port=DEVICE_PORT
    )

    # VIDEO CAPTURE CONFIGS
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # USER INTERFACE SETUP
    myGUI = interface.GUI(
                        play = True,
                        mode = "detection"
                        )

   # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = object_localization.Camera(
                camera_matrix=camera_matrix,
                camera_initial_position=calibration_position
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # GOAL POST DETECTION
    keypoint_regressor = object_localization.KeypointRegression()

    # OBJECT DETECTION MODEL
    PATH_TO_MODEL = cwd+"/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt"
    PATH_TO_LABELS = cwd+"/models/ssl_labels.txt"
    trt_net = object_detection.DetectNet(
                model_path = PATH_TO_MODEL, 
                labels_path = PATH_TO_LABELS, 
                input_width = 300, 
                input_height = 300,
                score_threshold = 0.5,
                draw = False,
                TRT_LOGGER = trt.Logger(trt.Logger.INFO)
                )
    trt_net.loadModel()

    # BALL TO PIXEL REGRESSION WEIGHTS
    regression_weights = np.loadtxt(cwd+"/models/regression_weights.txt")

    # CONFIGURING AND LOAD DURATION
    EXECUTION_TIME = 20
    config_time = time.time() - start
    print(f"Configuration Time: {config_time:.2f}s")
    avg_time = 0

    # START ROBOT INITIAL POSITION
    eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)

    # INIT VISION BLACKOUT STATE MACHINE
    state = "search"

    # INIT STATE MACHINE TIMER
    state_time = time.time()

    while cap.isOpened():
        start_time = time.time()
        
        current_frame = Frame(timestamp = time.time())
        if myGUI.play:
            ret, current_frame.input = cap.read()
            if not ret:
                print("Check video capture path")
                break
            else: myGUI.updateGUI(current_frame.input)

        detections = trt_net.inference(current_frame.input).detections

        for detection in detections:
            """
            Detection ID's:
            0: background
            1: ball
            2: goal
            3: robot

            Labels are available at: ssl-detector/models/ssl_labels.txt
            """
            class_id, score, xmin, xmax, ymin, ymax = detection               
            if class_id==2:
                # COMPUTE PIXEL FOR GOAL BOUNDING BOX -> USING BOTTOM CENTER FOR ALINGING
                left_corner, right_corner = keypoint_regressor.goalAsCorners(
                                                    src=current_frame.input,
                                                    left=xmin,
                                                    top=ymin,
                                                    right=xmax,
                                                    bottom=ymax)

                if keypoint_regressor.skip_frame == False:
                    # DRAW OBJECT POINTS ON SCREEN
                    if DRAW:
                        myGUI.drawCrossMarker(myGUI.screen, int(left_corner[0]), int(left_corner[1]))
                        myGUI.drawCrossMarker(myGUI.screen, int(right_corner[0]), int(right_corner[1]))
                    
                    # BACK PROJECT GOAL LEFT CORNER POSITION TO CAMERA 3D COORDINATES
                    left_corner_position = ssl_cam.pixelToCameraCoordinates(x=left_corner[0][0][0], y=left_corner[1][0][0], z_world=0)
                    left_corner_x, left_corner_y = left_corner_position[0], left_corner_position[1]

                    if DRAW:
                        caption = f"Position:{left_corner_x[0]:.2f},{left_corner_y[0]:.2f}"
                        myGUI.drawText(myGUI.screen, caption, (int(left_corner[0]-25), int(left_corner[1]+25)), 0.4)

                    # BACK PROJECT GOAL RIGHT CORNER POSITION TO CAMERA 3D COORDINATES
                    right_corner_position = ssl_cam.pixelToCameraCoordinates(x=right_corner[0][0][0], y=right_corner[1][0][0], z_world=0)
                    right_corner_x, right_corner_y = right_corner_position[0], right_corner_position[1]

                    if DRAW:
                        caption = f"Position:{right_corner_x[0]:.2f},{right_corner_y[0]:.2f}"
                        myGUI.drawText(myGUI.screen, caption, (int(right_corner[0]-25), int(right_corner[1]+25)), 0.4)
                    
                    # COMPUTE ROBOT RELOCALIZATION FROM GOAL CORNERS DETECTION
                    tx, ty, w = ssl_cam.selfLocalizationFromGoalCorners(
                            left_corner_x[0], 
                            left_corner_y[0], 
                            right_corner_x[0], 
                            right_corner_y[0])
                    cv2.imwrite("Goal Localization Test.jpg",myGUI.screen)
                    print(tx, ty, w)
                    # CONVERT COORDINATES FROM CAMERA TO ROBOT AXIS
                    tx, ty, _ = ssl_robot.cameraToRobotCoordinates(tx, ty)
                    #ssl_goal = current_frame.updateGoalCenter(x, y, score)

        # STATE MACHINE
        # TO-DO: move to state machine class
        if state == "search":
            target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
            if current_frame.has_goal: 
                state = "drive1"

        elif state == "drive1":
            target.type = communication_proto.pb.protoPositionSSL.target
            target.x = 0 - ssl_robot.x
            target.y = 0 - ssl_robot.y
            target.w = 0 - ssl_robot.w

        eth_comm.setPositionMessage(
                                x = target.x, 
                                y = target.y,
                                w = target.w,
                                posType = target.type)
        eth_comm.setKickMessage(
                            front=ssl_robot.front, 
                            charge=ssl_robot.charge, 
                            kickStrength=ssl_robot.kick_strength)
        eth_comm.sendSSLMessage()
        
        if state != "dock":
            eth_comm.resetRobotPosition()

        #print(f'State: {state} | Target: {target.x:.3f}, {target.y:.3f}, {target.w:.3f}, {target.type}')

        # DISPLAY WINDOW
        frame_time = time.time()-start_time
        avg_time = 0.8*avg_time + 0.2*frame_time
        if SHOW_DISPLAY:
            key = cv2.waitKey(10) & 0xFF
            quit = myGUI.commandHandler(key=key)   
            if DRAW: 
                myGUI.drawText(myGUI.screen, f"AVG FPS: {1/avg_time:.2f}s", (8, 13), 0.5)
            cv2.imshow(WINDOW_NAME, myGUI.screen)
            if quit:
                eth_comm.sendTargetPosition(x=0, y=0, w=0)
                break
        else:
            if time.time()-config_time-start>EXECUTION_TIME:
                print(f'Avg frame processing time:{avg_time}')
                break

    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
