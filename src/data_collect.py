import cv2
import os
import time
import csv
import numpy as np
import communication_proto
from ssl_vision_parser import SSLClient, FieldInformation
from object_detection import DetectNet

def serialize_data_to_log(frame_nr, ssl_vision_robot, ssl_vision_ball, jetson_vision_ball):
    # VISION BLACKOUT ROBOT POSITION
    id = ssl_vision_robot['id']
    robot_x = ssl_vision_robot['x']
    robot_y = ssl_vision_robot['y']
    robot_theta = ssl_vision_robot['orientation']

    # ON-FIELD BALL GROUND TRUTH POSITION
    ball_x = ssl_vision_ball['x']
    ball_y = ssl_vision_ball['y']    

    # DETECTED BALL BOUNDING BOX
    xmin, xmax, ymin, ymax = jetson_vision_ball

    data = [frame_nr, id, robot_x, robot_y, robot_theta, ball_x, ball_y,  xmin, xmax, ymin, ymax]
    return data


if __name__ == "__main__":
    cwd = os.getcwd()

    # START TIME
    start = time.time()
    EXECUTION_TIME = 300

    # UDP COMMUNICATION SETUP
    eth_comm = communication_proto.SocketUDP()

    # VIDEO CAPTURE CONFIGS
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # INIT OBJECT DETECTION 
    trt_net = DetectNet(
                model_path=cwd+"/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt", 
                labels_path=cwd+"/models/ssl_labels.txt", 
                score_threshold = 0.8,
                draw = False,
                display_fps = False
                )
    trt_net.loadModel()

    # INIT SSL CLIENT
    c = SSLClient()
    c.forceConnect(ip = '172.20.30.161', port = 10006)
    field = FieldInformation()

    # FRAME NR COUNTER
    frame_nr = 1

    # DATA FOR LOGS
    fields = ['FRAME_NR', 'ROBOT_ID', 'ROBOT X', 'ROBOT Y', 'ROBOT THETA', 'BALL X', 'BALL Y', 'X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX']
    CAMERA_ID = 2       # USING ONLY NEGATIVE Y CAMERA
    ROBOT_ID = 0
    data_log = []
    lastHasBall = False

    while True:
        # INIT DETECTION OBJECTS
        ssl_vision_robot, ssl_vision_ball, jetson_vision_ball = None, None, None

        # RECEIVE MSG FROM MCU
        odometry, hasBall, kickLoad, battery, count = eth_comm.recvSSLMessage()

        # RECEIVE DETECTIONS FROM SSL VISION
        ret, pkg = c.receive()
        if ret:
            field.update(pkg.detection)
            if pkg.detection.camera_id==CAMERA_ID:
                f = field.getAll(CAMERA_ID)
                for robot in f['yellowRobots']:
                    if robot['id'] == ROBOT_ID:
                        ssl_vision_robot = robot
                for ball in f['balls']:
                    ssl_vision_ball = ball

        # CAPTURE FRAME
        if cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               print("Check video capture path")
               break
           else: img = frame.copy()
        
        # DETECT OBJECT'S BOUNDING BOXES
        detections = trt_net.inference(img).detections
        for detection in detections:
            class_id, score, xmin, xmax, ymin, ymax = detection
            if class_id==1:
                jetson_vision_ball = [xmin, xmax, ymin, ymax]

        # CHECK IF ALL FIELDS ARE AVAILABLE
        valid = (ssl_vision_robot is not None) and (ssl_vision_ball is not None) and (jetson_vision_ball is not None)

        # PRINT FOR DEBUG
        if valid:
            data = serialize_data_to_log(frame_nr, ssl_vision_robot, ssl_vision_ball, jetson_vision_ball)
            print(f"hasBall: {hasBall} | robot: {ssl_vision_robot} | ball: {ssl_vision_ball}, {jetson_vision_ball}")

        # ADD DETECTIONS TO LOG IF ROBOT HAS BALL ON SENSOR
        if valid and hasBall and not lastHasBall:
            # SAVE RAW FRAME
            dir = cwd+f"/data/object_localization/{frame_nr}.jpg"
            cv2.imwrite(dir, frame)

            # APPEND DATA TO LOG
            data_log.append(data)

            # ADD FRAME NR
            frame_nr += 1
        lastHasBall = hasBall

        # FINISH AND SAVE LOG IF ROBOT IS TURNED OFF (OR RESET)
        if battery<14:
            break

    dir = cwd+f"/data/object_localization/log.csv"
    with open(dir, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(data_log)

    cap.release()
    cv2.destroyAllWindows()
