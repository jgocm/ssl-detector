import cv2
import numpy as np
import tensorrt as trt
import time
import socket
import argparse
import sys
import os

# LOCAL IMPORTS
import robot
import object_detection
import object_localization
import communication_proto
import interface

def main():
    # START TIME
    start = time.time()

    # DISPLAY TITLE
    WINDOW_NAME = 'Vision Blackout'
    SHOW_DISPLAY = True

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
    PATH_TO_INTRINSIC_PARAMETERS = "/home/joao/ssl-detector/configs/camera_matrix_C922.txt"
    PATH_TO_DISTORTION_PARAMETERS = "/home/joao/ssl-detector/configs/camera_distortion_C922.txt"
    PATH_TO_2D_POINTS = "/home/joao/ssl-detector/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = "/home/joao/ssl-detector/configs/calibration_points3d.txt"
    ssl_cam = object_localization.Camera(
                camera_matrix_path=PATH_TO_INTRINSIC_PARAMETERS,
                camera_distortion_path=PATH_TO_DISTORTION_PARAMETERS
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # OBJECT DETECTION MODEL
    PATH_TO_MODEL = "/home/joao/ssl-detector/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt"
    PATH_TO_LABELS = "/home/joao/ssl-detector/models/ssl_labels.txt"
    trt_net = object_detection.DetectNet(
                model_path = PATH_TO_MODEL, 
                labels_path = PATH_TO_LABELS, 
                input_width = 300, 
                input_height = 300,
                score_threshold = 0.5,
                draw = True,
                TRT_LOGGER = trt.Logger(trt.Logger.INFO)
                )
    trt_net.loadModel()

    # CONFIGURING AND LOAD DURATION
    config_time = time.time()- start
    print(f"Configuration Time: {config_time:.2f}s")

    while cap.isOpened():
        if myGUI.play:
            ret, frame = cap.read()
            if not ret:
                print("Check video capture path")
                break
            else: myGUI.updateGUI(frame)

        detections = trt_net.inference(frame).detections

        for detection in detections:
            class_id, score, xmin, xmax, ymin, ymax = detection
            if class_id==1:     # ball
                #print("ball detected")
                # COMPUTE PIXEL FOR BALL POSITION
                pixel_x, pixel_y = ssl_cam.ballAsPoint(left=xmin, top=ymin, right=xmax, bottom=ymax, weight_y=0.2)
            
                # DRAW OBJECT POINT ON SCREEN
                myGUI.drawCrossMarker(myGUI.screen, int(pixel_x), int(pixel_y))

                # BACK PROJECT BALL POSITION TO CAMERA 3D COORDINATES
                object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
                x, y = object_position[0], object_position[1]

                caption = f"Position:{x[0]:.2f},{y[0]:.2f}"
                myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+25)), 0.4)

                # CONVERT COORDINATES FROM CAMERA TO ROBOT AXIS
                x, y = ssl_robot.cameraToRobotCoordinates(x, y)
                x = -x[0]/1000
                y = -y[0]/1000

                # SEND OBJECT RELATIVE POSITION TO ROBOT THROUGH ETHERNET CABLE w/ SOCKET UDP
                eth_comm.sendPosition(x=x, y=y, w=0)
                
        # DISPLAY WINDOW
        if SHOW_DISPLAY:
            key = cv2.waitKey(10) & 0xFF
            quit = myGUI.commandHandler(key=key)
            run_time = time.time()-start
            myGUI.drawText(myGUI.screen, f"running time: {run_time:.2f}s", (8, 13), 0.5)
            cv2.imshow(WINDOW_NAME, myGUI.screen)
            if quit:
                eth_comm.sendPosition(x=0, y=0, w=0)
                break
        else:
            if time.time()-config_time>60:
                break

        
    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()