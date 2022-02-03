from base64 import encode
import cv2
from robot import robot
from object_localization import camera
from object_detection import detectnet
from GUI import GUI
import udp_send

import jetson.inference
import jetson.utils

import numpy as np
import time
import socket
import argparse
import sys
import os

def loadNetwork():
    # network args
    ssl_mb2_model='/home/joao/ssl-detector/models/ssl3/mb2-ssd-lite.onnx'
    ssl_labels='/home/joao/ssl-detector/models/labels.txt'
    batch_size =4
    input_blob='--input-blob=input_0'
    output_cvg='--output-cvg=scores'
    output_bbox='--output-bbox=boxes'

    net = detectnet(network="ssd-mobilenet-v2",
                    path_to_model=ssl_mb2_model,
                    path_to_labels=ssl_labels,
                    batch_size=batch_size,
                    ).net
    
    return net

def goalCoordinates():
    goal_height = 65
    goal_length = 700
    x0, y0, z0 = -350,3000,0
    left_top = (x0,y0,z0+goal_height)
    right_top = (x0+goal_length,y0,z0+goal_height)
    left_bottom = (x0,y0,z0)
    right_bottom = (x0+goal_length,y0,z0)
    goal3dCoordinates = np.array([
                    left_top,
                    right_top,
                    left_bottom,
                    right_bottom,
                    ],dtype="float64")  
    return goal3dCoordinates

def fieldCoordinates():
    # Coordinates system parallel to camera vision axis:
    # Center as origin
    # y axis = center to goal
    # x axis = left to right
    dx = -2000
    dy = +3000
    w1 = (0+dx,0+dy,0)
    w2 = (1000+dx,0+dy,0)
    w3 = (1650+dx,0+dy,0)
    w4 = (2000+dx,0+dy,0)
    w5 = (2350+dx,0+dy,0)
    w6 = (3000+dx,0+dy,0)
    w7 = (4000+dx,0+dy,0)
    w8 = (1000+dx,-800+dy,0)
    w9 = (3000+dx,-800+dy,0)
    w10 = (2000+dx,-3000+dy,0)

    field_points = np.array([
                        w1,w2,w3,w4,w5,w6,w7,w8,w9,w10
                        ],dtype="float64")
    return field_points

def getFPS(lastTimeStamp, lastFps):
    dt=time.time()-lastTimeStamp
    fps=1/dt
    fps=.7*lastFps + .3*fps
    return fps

#### MAIN CODE ####

# V4L2 camera path = "/dev/video0"
# CSI camera path = "csi://0"
camera_path = "/dev/video0"     

# PS Eye camera parameters
mtx = np.array([
                (522.572,0,331.090),
                (0,524.896,244.689),
                (0,0,1)
                ])
#camera_height = 320.00
#camera_height = 180.00
camera_height = 207.49
# create object detection network 
net = loadNetwork()

# create video output object
output = jetson.utils.videoOutput("display://0")

# OPENCV VIDEO SOURCE AND OUTPUT
input = cv2.VideoCapture(camera_path)
dispW = 1680
dispH = 1050
input.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
input.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

# instantiate robot
ssl_robot = robot()

# instantiate camera vision
vision = camera(
    camera_path=camera_path,
    camera_matrix=mtx,
    camera_height=camera_height
)

# capture the next image
ret, frame = input.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float32)

img = jetson.utils.cudaFromNumpy(img)

# detect objects in the image
detections = net.Detect(img, dispW, dispH)

img = jetson.utils.cudaToNumpy(img, dispW, dispH, 4)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# process frames until the user exits
play = True
fps=0
winname = "SSL Detector"

# user interface
play = True
mode = 'detection'
font = cv2.FONT_HERSHEY_SIMPLEX
controller = GUI(img=img, play=play, display_menu=False, mode=mode)

# experiment number
exp_nr = 69
load_nr = 47

# communication
UDP_IP ='172.20.10.2'
#UDP_IP ='192.168.1.7'
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET,
                    socket.SOCK_DGRAM)


while input.isOpened():
    timeStamp = time.time()

    if controller.play==True:
        # capture the next image
        ret, frame = input.read()
        #frame = cv2.imread('experiments/42.jpg')
        frame = cv2.imread('48.jpg')
        #frame = cv2.imread('/home/joao/ssl-dataset/1_resized/00285.jpg')
        last_frame = frame

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float32)

        img = jetson.utils.cudaFromNumpy(img)

        if mode=='detection':
            # detect objects in the image
            detections = net.Detect(img, dispW, dispH)

        img = jetson.utils.cudaToNumpy(img, dispW, dispH, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        fps = round(getFPS(lastTimeStamp=timeStamp, lastFps=fps),2)
        #window_title = "SSL Detector | Network {:.0f} FPS".format(fps)

    elif controller.play==False:
        img = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGBA).astype(np.float32)

        img = jetson.utils.cudaFromNumpy(img)

        if controller.mode=='detection':
            # detect objects in the image
            detections = net.Detect(img, dispW, dispH)

        img = jetson.utils.cudaToNumpy(img, dispW, dispH, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    controller.updateImg(img)
    line_x = 320
    line_y = 240
    cv2.line(img, (line_x,0), (line_x,2*line_y), (0,0,0), 1)
    cv2.line(img, (0,line_y), (2*line_x,line_y), (0,0,0), 1)

    if controller.mode=='detection':
        ballToCamera,robotToCamera,goalToCamera = [0,0,0],[0,0,0],[0,0,0]
        for detection in detections:
            ID = detection.ClassID			# class ID number
            item = net.GetClassDesc(ID)		# class ID title
            confidence = detection.Confidence
            top = int(detection.Top)		# top position
            bottom = int(detection.Bottom)	# bottom position
            left = int(detection.Left)		# left position
            right = int(detection.Right)	# right position
            #controller.objToMark=item
            if controller.objToDetect[0]:
                if item=="goal":      
                    points2d = controller.goalAsPoints(left=left, top=top, right=right, bottom=bottom)
                    points3d = goalCoordinates()
                    #left_top = points2d[0]  
                    #right_top = points2d[1]
                    left_bottom = points2d[2]
                    right_bottom = points2d[3]
                    center_bottom = (left_bottom+right_bottom)/2
                    flag = ssl_robot.isLocated()
                    flag = False
                    color=(0,255,0)
                    pointToCamera = vision.pixelToCameraCoordinates(x=center_bottom[0],y=center_bottom[1])
                    goalToCamera = pointToCamera
                    pointToField = pointToCamera
                    if flag:
                        pointToField = vision.pixelToWorldCoordinates(x=center_bottom[0],y=center_bottom[1],height=0)
                    controller.drawDetectionMarker(item=item,
                                            color=color,
                                            point=center_bottom, 
                                            pointToCamera=pointToCamera, 
                                            pointToField=pointToField, 
                                            flag=flag)

                    pointToCamera = vision.pixelToCameraCoordinates(x=left_bottom[0],y=left_bottom[1])
                    if flag:
                        pointToField = vision.pixelToWorldCoordinates(x=left_bottom[0],y=left_bottom[1],height=0)
                    '''controller.drawDetectionMarker(item=item,
                                            color=color,
                                            point=left_bottom, 
                                            pointToCamera=pointToCamera, 
                                            pointToField=pointToField, 
                                            flag=flag)'''

                    pointToCamera = vision.pixelToCameraCoordinates(x=right_bottom[0],y=right_bottom[1])
                    if flag:
                        pointToField = vision.pixelToWorldCoordinates(x=right_bottom[0],y=right_bottom[1],height=0)
                    '''controller.drawDetectionMarker(item=item,
                                            color=color,
                                            point=right_bottom, 
                                            pointToCamera=pointToCamera, 
                                            pointToField=pointToField, 
                                            flag=flag)'''
                    
                    position = vision.computePoseToBBox(left=left, top=top, right=right, bottom=bottom)
                    #vision.setPose(points3d, points2d)
                    position, rotation = vision.position, vision.rotation
                    ssl_robot.setPose(position=position, euler_angles=rotation)
                    print("Robot position:")
                    print(ssl_robot.position)

            if controller.objToDetect[1]:
                if item=="ball":
                    ballAsPoint = controller.ballAsPoint(left=left, top=top, right=right, bottom=bottom)
                    ballToCamera = vision.pixelToCameraCoordinates(x=ballAsPoint[0],y=ballAsPoint[1])
                    flag = ssl_robot.isLocated()
                    flag = False
                    ballToField = None
                    color=(255,0,0)
                    if flag:
                        ballToField = vision.pixelToWorldCoordinates(x=ballAsPoint[0],y=ballAsPoint[1],height=0)
                    
                    controller.drawDetectionMarker(item=item,
                                            color=color,
                                            point=ballAsPoint, 
                                            pointToCamera=ballToCamera, 
                                            pointToField=ballToField, 
                                            flag=flag)

            if controller.objToDetect[2]:
                if item=="robot":
                    print("ROBOT DETECTED")
                    print("Robot to Camera Relative Position:")
                    robotAsPoint=controller.robotAsPoint(left=left, top=top, right=right, bottom=bottom)
                    robotToCamera = vision.pixelToCameraCoordinates(x=robotAsPoint[0],y=robotAsPoint[1])
                    flag = ssl_robot.isLocated()
                    robotToField = robotToCamera
                    color=(0,0,255)
                    flag = True
                    flag = False
                    if flag:
                        robotToField = vision.pixelToWorldCoordinates(x=robotAsPoint[0],y=robotAsPoint[1],height=0)

                    controller.drawDetectionMarker(item=item,
                                            color=color,
                                            point=robotAsPoint, 
                                            pointToCamera=robotToCamera, 
                                            pointToField=robotToField, 
                                            flag=flag)

        ball_data,robot_data,goal_data = udp_send.shapeData(ballToCamera,robotToCamera,goalToCamera)                                   
        data = udp_send.encodePacket(controller.objToDetect,ball_data,robot_data,goal_data)
        #print(data)
        data = udp_send.int_to_bytes(data)
        sock.sendto(data, (UDP_IP, UDP_PORT))
    if controller.mode=='marker':
        for point in controller.marker_points:
            x, y = point[0]
            color = point[1]
            pointToCamera = vision.pixelToCameraCoordinates(x=x,y=y)
            #flag = ssl_robot.isLocated()
            pointToField = None
            flag = False
            if flag:
                pointToField = vision.pixelToWorldCoordinates(x=x,y=y,height=0)

            controller.drawMarkerPoint(point=point, 
                                    pointToCamera=pointToCamera, 
                                    pointToField=pointToField, 
                                    flag=flag)

    if controller.mode=='calibration':
        if controller.isCollecting()==False:
            field_points = fieldCoordinates()
            points2d=[]
            points3d=[]
            for i in range(0,len(controller.marker_points)):
                p = controller.marker_points[i]
                if p[0][0]>0:
                    points2d.append(p[0])
                    points3d.append(field_points[i])
            points2d = np.array(points2d,dtype="float64")
            points3d = np.array(points3d,dtype="float64")

            # FIND CAMERA POSE
            vision.setPose(points3d, points2d)
            position, rotation = vision.position, vision.rotation
            ssl_robot.setPose(position, rotation)
            print("Robot position:")
            print(ssl_robot.position)

    img = controller.updateGUI(img)
    
    cv2.imshow(winname, img)
    key=cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF  == ord(' '):
        key=cv2.waitKey(0)
        if key & 0xFF == ord(' '):
            continue
    elif key & 0xFF == ord('s'):
        cv2.imwrite(f'{exp_nr}.jpg',img)
        if controller.mode=='calibration':
            cv2.imwrite(f'experiments/{exp_nr}.jpg',img)
            np.savetxt(f'experiments/{exp_nr}_camera_rotation.txt',vision.calib_rotation)
            np.savetxt(f'experiments/{exp_nr}_camera_position.txt',vision.calib_position)
            np.savetxt(f'experiments/{exp_nr}_points2d.txt',points2d)
            np.savetxt(f'experiments/{exp_nr}_points3d.txt',points3d)
            print(f'EXPERIMENT {exp_nr} DATA SAVED')
        print('IMAGE SAVED')
        exp_nr+=1
    elif key & 0xFF == ord('l'):
        nr=load_nr
        points2d = np.loadtxt(f'experiments/{nr}_points2d.txt', dtype="float64")
        points3d = np.loadtxt(f'experiments/{nr}_points3d.txt', dtype="float64")
        # FIND CAMERA POSE
        vision.setPose(points3d, points2d)
        position, rotation = vision.position, vision.rotation
        ssl_robot.setPose(position, rotation)
        print("Robot position:")
        print(ssl_robot.position)
    else:
        controller.commandHandler(key=key)

    print(controller.state)
    #print(controller.objToDetect)
    
    cv2.setMouseCallback(winname,controller.markPoint)

input.release()
cv2.destroyAllWindows()