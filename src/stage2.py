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
from entities import robot
import object_detection
import object_localization
import communication_proto
import interface

def offsetTarget(x, y, w, offset=1):
    dist = math.sqrt(x**2+y**2) + 0.001
    prop = (dist - offset)/dist
    target_x, target_y, target_w = prop*x, prop*y, w
    return target_x, target_y, target_w

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

    # DISPLAYS POSITIONS AND MARKERS ON SCREEN
    DRAW = False

    # DISPLAY TITLE
    WINDOW_NAME = 'Vision Blackout'
    SHOW_DISPLAY = False

    # ROBOT SETUP
    ROBOT_ID = 0
    ROBOT_HEIGHT = 155
    ROBOT_DIAMETER = 180
    CAMERA_TO_CENTER_OFFSET = 90
    ssl_robot = robot.SSLRobot(                
                id = ROBOT_ID,
                height = ROBOT_HEIGHT,
                diameter = ROBOT_DIAMETER,
                camera_offset = CAMERA_TO_CENTER_OFFSET
                )

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
    config_time = time.time() - start
    print(f"Configuration Time: {config_time:.2f}s")
    avg_time = 0

    # START ROBOT INITIAL POSITION
    eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)

    # INIT MSG
    target_x, target_y, target_w = 0,0,0
    front, charge, kickStrength = False, True, 40

    # INIT VISION BLACKOUT STATE MACHINE
    state = "search"

    # INIT STATE MACHINE TIMER
    state_time = time.time()

    while cap.isOpened():
        HAS_BALL = False
        HAS_GOAL = False
        start_time = time.time()

        if myGUI.play:
            ret, frame = cap.read()
            if not ret:
                print("Check video capture path")
                break
            else: myGUI.updateGUI(frame)

        detections = trt_net.inference(frame).detections

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
            if class_id==1:     # ball
                # COMPUTE PIXEL FOR BALL POSITION
                pixel_x, pixel_y = ssl_cam.ballAsPointLinearRegression(
                                                                    left=xmin, 
                                                                    top=ymin, 
                                                                    right=xmax, 
                                                                    bottom=ymax, 
                                                                    weight_x=regression_weights[0],
                                                                    weight_y=regression_weights[1])
            
                # DRAW OBJECT POINT ON SCREEN
                if DRAW:
                    myGUI.drawCrossMarker(myGUI.screen, int(pixel_x), int(pixel_y))

                # BACK PROJECT BALL POSITION TO CAMERA 3D COORDINATES
                object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
                x, y = object_position[0], object_position[1]

                if DRAW:
                    caption = f"Position:{x[0]:.2f},{y[0]:.2f}"
                    myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+25)), 0.4)

                # CONVERT COORDINATES FROM CAMERA TO ROBOT AXIS
                x, y, w = ssl_robot.cameraToRobotCoordinates(x[0], y[0])
                ball_x, ball_y, ball_w = x, y, w

                HAS_BALL = True
              
            if class_id==2:
                # COMPUTE PIXEL FOR GOAL BOUNDING BOX -> USING BOTTOM CENTER FOR ALINGING
                pixel_x, pixel_y = ssl_cam.goalAsPoint(
                                                    left=xmin,
                                                    top=ymin,
                                                    right=xmax,
                                                    bottom=ymax)
                # DRAW OBJECT POINT ON SCREEN
                if DRAW:
                    myGUI.drawCrossMarker(myGUI.screen, int(pixel_x), int(pixel_y))
                
                # BACK PROJECT GOAL CENTER POSITION TO CAMERA 3D COORDINATES
                object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
                x, y = object_position[0], object_position[1]

                if DRAW:
                    caption = f"Position:{x[0]:.2f},{y[0]:.2f}"
                    myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+25)), 0.4)
                
                # CONVERT COORDINATES FROM CAMERA TO ROBOT AXIS
                x, y, w = ssl_robot.cameraToRobotCoordinates(x[0], y[0])
                goal_x, goal_y, goal_w = x, y, w

                HAS_GOAL = True

        # STATE MACHINE
        # TO-DO: create state machine class
        if state == "search":
            eth_comm.sendRotateSearch()
            if HAS_BALL: 
                state = "drive1"
                eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)
        elif state == "drive1":
            target_x, target_y, target_w = offsetTarget(ball_x, ball_y, ball_w, 0.5)
            eth_comm.sendTargetPosition(x=target_x, y=target_y, w=target_w)
            dist = math.sqrt(target_x**2+target_y**2)+0.001
            if dist<0.05:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                state = "stop1"
                state_time = time.time()
                eth_comm.sendStopMotion()
        elif state == "stop1":
            eth_comm.sendStopMotion()
            if dist>0.05:
                state = "drive1"
                eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)
            elif time.time()-state_time>0.5:
                state = "align1"
        elif state ==  "align1":
            target_x, target_y, target_w = ball_x, ball_y, ball_w
            eth_comm.sendRotateControl(x=target_x, y=target_y, w=target_w)
            if np.abs(target_w)<0.125:
                state = "rotate"
                state_time = time.time()
        elif state == "rotate":
            eth_comm.sendRotateInPoint()
            if HAS_BALL and HAS_GOAL:
                target_x, target_y, target_w = directionVector(goal_x, goal_y, ball_x, ball_y)
                print(f"w = {target_w:.3f}")
                if np.abs(target_w) < 0.3:
                    state = "stop2"
                    state_time = time.time()
                    eth_comm.sendStopMotion()
        elif state == "stop2":
            eth_comm.sendStopMotion()
            if time.time()-state_time > 0.5:
                state = "drive2"
                eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)
        elif state == "drive2":
            vx, vy, target_w = directionVector(goal_x, goal_y, ball_x, ball_y)
            target_x, target_y, _ = alignedTarget(vx, vy, ball_x, ball_y, offset=0.3)
            #target_x = target_x-ssl_robot.camera_offset/1000 # robot center to camera distance correction
            eth_comm.sendTargetPosition(target_x, target_y, target_w)
            dist = math.sqrt(target_x**2+target_y**2)+0.001
            if np.abs(target_w)<0.05 and dist<0.07:
                state = "stop3"
                state_time = time.time()
                eth_comm.sendStopMotion() 
        elif state == "stop3":
            eth_comm.sendStopMotion() 
            if time.time()-state_time > 0.5:
                state = "align2"
                eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)
        elif state == "align2":
            vx, vy, target_w = directionVector(goal_x, goal_y, ball_x, ball_y)
            target_x, target_y = ball_x, ball_y
            eth_comm.sendRotateControl(x=target_x, y=target_y, w=target_w)
            if np.abs(target_w)<0.125:
                state = "drive3"
                eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)
        elif state == "drive3":
            vx, vy, target_w = directionVector(goal_x, goal_y, ball_x, ball_y)
            target_x, target_y, target_w = ball_x-ssl_robot.camera_offset/1000, ball_y, 0
            eth_comm.sendTargetPosition(x=target_x, y=target_y, w=target_w)
            dist = math.sqrt(target_x**2+target_y**2)+0.001
            if dist<0.270 and not HAS_BALL:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    state = "dock"
                    state_time = time.time()
        elif state == "dock":
            target_x = ball_x
            front, charge, kickStrength = True, False, 40
            if time.time()-state_time > 3:
                charge = True
            if time.time()-state_time > 3.1:
                eth_comm.sendStopMotion() 
                break
            eth_comm.setKickMessage(front=front, charge=charge, kickStrength=kickStrength)
            eth_comm.sendBallDocking(x=target_x, y=target_y, w=target_w)
        
        print(f'State: {state} | Target: {target_x:.3f}, {target_y:.3f}, {target_w:.3f}')

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
            if time.time()-config_time-start>60:
                print(f'Avg frame processing time:{avg_time}')
                break

    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
