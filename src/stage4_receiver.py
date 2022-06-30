
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
from fsm import FSM, Stage2States
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
    SHOW_DISPLAY = False

    # DISPLAYS POSITIONS AND MARKERS ON SCREEN
    DRAW = SHOW_DISPLAY

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
    ssl_robot_allie = Robot()
    target = TargetPoint(x = 0, y = 0, w = 0)

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

    # INIT VISION BLACKOUT STATE MACHINE
    INITIAL_STATE = 1
    state_machine = FSM(
        stage = 2,
        initial_state = INITIAL_STATE,
        init_time = start)


    # CONFIGURING AND LOAD DURATION
    EXECUTION_TIME = 120
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
            ret, frame = cap.read()
            if not ret:
                print("Check video capture path")
                break
            else: myGUI.updateGUI(frame)

        detections = trt_net.inference(myGUI.screen).detections

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
                pixel_x, pixel_y = ssl_cam.ballAsPoint(
                                                    left=xmin, 
                                                    top=ymin, 
                                                    right=xmax, 
                                                    bottom=ymax)

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
                ssl_ball = current_frame.updateBall(x, y, score)
              
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
                ssl_goal = current_frame.updateGoalCenter(x, y, score)

            if class_id==3:
                #COMPUTE PIXEL FOR ROBOT ASSISTANT POSITION
                pixel_x, pixel_y = ssl_cam.robotAsPoint(
                                                    left=xmin,
                                                    top=ymin,
                                                    right=xmax,
                                                    bottom=ymax
                )
                #DRAW OBJECT PONIT ON SCREEN
                if DRAW:
                    myGUI.drawCrossMarker(myGUI.screen, int(pixel_x), int(pixel_y))
                
                # BACK PROJECT ROBOT POSITION TO CAMERA 3D COORDINATES
                object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
                x, y = object_position[0], object_position[1]

                if DRAW:
                    caption = f"Position:{x[0]:.2f},{y[0]:.2f}"
                    myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+25)), 0.4)

                # CONVERT COORDINATES FROM CAMERA TO ROBOT AXIS
                x, y, w = ssl_robot.cameraToRobotCoordinates(x[0], y[0])
                ssl_robot_allie = current_frame.updateRobot(x, y, score)
              


        # STATE MACHINE
        # TO-DO: create state machine class
        if state == "search":
            target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
            if current_frame.has_ball: 
                state = "drive1"

        elif state == "drive1":
            target.type = communication_proto.pb.protoPositionSSL.target
            target.x, target.y = offsetTarget(ssl_ball.x, ssl_ball.y)
            target.w = ssl_ball.w
            dist = math.sqrt(target.x**2+target.y**2)+0.001
            if dist<0.1:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                state = "stop1"
                state_time = time.time()
                target.type = communication_proto.pb.protoPositionSSL.stop

        elif state == "stop1":
            target.type = communication_proto.pb.protoPositionSSL.stop
            if dist>0.1:
                state = "drive1"
            elif time.time()-state_time>0.5:
                state = "align"

        elif state ==  "align":
            target.type = communication_proto.pb.protoPositionSSL.rotateControl
            target.x, target.y, target.w = ssl_ball.x, ssl_ball.y, ssl_ball.w
            if np.abs(target.w)<=0.050:
                state = "waitToReceive"
                state_time = time.time()

        elif state == "waitToReceive":
            target.type = communication_proto.pb.protoPositionSSL.rotateControl
            target.x, target.y, target.w = ssl_ball.x, ssl_ball.y, ssl_ball.w
            dist = math.sqrt(target.x**2+target.y**2)+0.001
            if dist<0.4:
                state = "prepareToKick"
                state_time = time.time()
                
        elif state == "prepareToKick":
            target.type = communication_proto.pb.protoPositionSSL.rotateControl
            if time.time()-state_time>4:
                state = "backwardsForSearch"
                state_time = time.time()

        elif state == "backwardsForSearch":
            target.type = communication_proto.pb.protoPositionSSL.dock
            target.x, target.y, target.w = -1, 0, 0
            target.reset_odometry = False
            if time.time()-state_time>3:
                state="setGoal"
      
        elif state == "setGoal":
            # STATE MACHINE
            ssl_robot.charge = True
            target, ssl_robot = state_machine.stage2(
                                    frame = current_frame, 
                                    ball = ssl_ball,
                                    goal = ssl_goal,
                                    robot = ssl_robot)
        
            # UPDATE PROTO MESSAGE
            eth_comm.setSSLMessage(target = target, robot = ssl_robot)            

        if state != "setGoal":
            eth_comm.setPositionMessage(
                                    x = target.x, 
                                    y = target.y,
                                    w = target.w,
                                    pos_type = target.type,
                                    reset_odometry = target.reset_odometry)                        
            eth_comm.sendSSLMessage()       
            

            print(f'State: {state} | Target: {target.x:.3f}, {target.y:.3f}, {target.w:.3f}, {target.type}, {target.reset_odometry}')
        else:
            # ACTION
            eth_comm.sendSSLMessage()
            print(f'{state_machine.current_state} | Target: {eth_comm.msg.x:.3f}, {eth_comm.msg.y:.3f}, {eth_comm.msg.w:.3f}, {eth_comm.msg.posType}, {eth_comm.msg.resetOdometry}')
    
            if state_machine.current_state == Stage2States.finish and state_machine.getStateDuration(current_timestamp=current_frame.timestamp)>1:
                break

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

