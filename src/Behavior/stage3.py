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
from Entities.entities import Robot, Goal, Ball, Frame, Field
import Vision.object_detection as object_detection
import Vision.camera_transformation as camera_transformation
import Communication.communication_proto as communication_proto
import Calibration.interface as interface
from Navigation.navigation import GroundPoint, TargetPoint

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
    INITIAL_POSE = -3.474, 2.223, 0
    ssl_robot = Robot(                
                id = ROBOT_ID,
                height = ROBOT_HEIGHT,
                diameter = ROBOT_DIAMETER,
                camera_offset = CAMERA_TO_CENTER_OFFSET,
                initial_pose = INITIAL_POSE
                )

    # INIT ENTITIES
    ssl_field = Field()
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
    ssl_cam = camera_transformation.Camera(
                camera_matrix=camera_matrix,
                camera_initial_position=calibration_position
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # GOAL POST DETECTION
    keypoint_regressor = camera_transformation.KeypointRegression()

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

    # BALL TO PIXEL REGRESSION WEIGHTS
    regression_weights = np.loadtxt(cwd+"/models/regression_weights.txt")

    # CONFIGURING AND LOAD DURATION
    EXECUTION_TIME = 180
    config_time = time.time() - start
    print(f"Configuration Time: {config_time:.2f}s")
    avg_time = 0

    # START ROBOT INITIAL POSITION
    eth_comm.sendSourcePosition(x = 0, y = 0, w = 0)

    # INIT VISION BLACKOUT STATE MACHINE
    state = "search"

    # INIT STATE MACHINE TIMER
    STAGE = 3
    state_time = time.time()

    # STAGE 3 TARGET POINT
    FINAL_X = 1.311
    FINAL_Y = -0.731

    RELOCALIZATION_RADIUS = 1.5

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

                if keypoint_regressor.skip_frame == False:
                    coef, intercept = keypoint_regressor.goalAsLine(
                                    src=current_frame.input,
                                    left=xmin,
                                    top=ymin,
                                    right=xmax,
                                    bottom=ymax)
                    if keypoint_regressor.skip_frame:
                        print("Line regression has failed")
                    else:
                        w = ssl_cam.selfOrientationFromGoalLine(
                                                coef,
                                                intercept)
                        R = math.sqrt(ssl_goal.center_x**2 + ssl_goal.center_y**2)
                        tx = (ssl_field.length/2 - R*math.cos(w))
                        ty = -R*math.sin(w)
                        ssl_robot.updateSelfPose(tx, ty, w)

        # STATE MACHINE
        # TO-DO: move to state machine class
        if state == "search":
            target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
            if current_frame.has_goal: 
                state = "driveTowardsGoalCenter"

            goal_distance = math.sqrt(ssl_goal.center_x**2 + ssl_goal.center_y**2)
            left_goal_distance = math.sqrt((ssl_field.left_goal.center_x-ssl_robot.tx)**2 + (ssl_field.left_goal.center_y-ssl_robot.ty)**2)
            right_goal_distance = math.sqrt((ssl_field.right_goal.center_x-ssl_robot.tx)**2 + (ssl_field.right_goal.center_y-ssl_robot.ty)**2)

            if np.abs(goal_distance - left_goal_distance) < np.abs(goal_distance - right_goal_distance):
                FINAL_X = -FINAL_X
                FINAL_Y = -FINAL_Y

        elif state == "driveTowardsGoalCenter":
            target.type = communication_proto.pb.protoPositionSSL.dock
            target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                x1=ssl_robot.x,
                y1=ssl_robot.y,
                x2=ssl_goal.center_x,
                y2=ssl_goal.center_y,
                relative_angle=0,
                relative_distance=-RELOCALIZATION_RADIUS)

            if target.getDistance() < 0.05:
                state = "alignToGoalCenter"
                state_time = time.time()

        elif state == "alignToGoalCenter":
            target.type = communication_proto.pb.protoPositionSSL.rotateControl
            target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                x1=0,
                y1=0,
                x2=ssl_goal.center_x,
                y2=ssl_goal.center_y,
                relative_angle=0,
                relative_distance=-RELOCALIZATION_RADIUS
            )          
            if np.abs(target.w) < 0.1:
                state = "stopAndRelocalize"
                state_time = time.time()
                ssl_robot.is_located = False

        elif state == "stopAndRelocalize":
            target.type = communication_proto.pb.protoPositionSSL.stop
            keypoint_regressor.skip_frame = False
            target.reset_odometry = True
            if time.time() - state_time > 1:
                keypoint_regressor.skip_frame = True
                if ssl_robot.is_located:
                    target.x, target.y, target.w = ssl_robot.tx, ssl_robot.ty, ssl_robot.w
                    state = "rotateToTargetDirection"
                    state_time = time.time()
                else:
                    state = "search"
        
        elif state == "rotateToTargetDirection":
            target.type = communication_proto.pb.protoPositionSSL.rotateControl
            target.x, target.y = FINAL_X - ssl_robot.tx, FINAL_Y - ssl_robot.ty
            target.w = (target.getDirection() - ssl_robot.w)
            target.reset_odometry = False
            print(target.getDirection())
            if time.time() - state_time > 2:
                state = "driveToTargetPosition"
                state_time = time.time()
                target.reset_odometry = True

        elif state == "driveToTargetPosition":
            target.type = communication_proto.pb.protoPositionSSL.target
            dist = math.sqrt((FINAL_X - ssl_robot.tx)**2 + (FINAL_Y - ssl_robot.ty)**2)
            target.x, target.y, target.w = dist, 0, 0
            target.reset_odometry = False

            if time.time() - state_time > 10:
                state = "finish"
                state_time = time.time()

        elif state == "finish":
            target.type = communication_proto.pb.protoPositionSSL.stop
            target.reset_odometry = True
            if time.time() - state_time > 1:
                break

        eth_comm.setSSLMessage(target = target, robot = ssl_robot)
        eth_comm.sendSSLMessage()

        print(f'State: {state} | Target: {target.x:.3f}, {target.y:.3f}, {target.w:.3f}, {target.type}, {target.reset_odometry}')

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
