import cv2
import os
import time
import csv
import numpy as np
import communication_proto
from ssl_vision_parser import SSLClient, FieldInformation
from object_detection import DetectNet
from jetson_vision import JetsonVision
import plot
import tensorrt as trt

def serialize_data_to_log(frame_nr, id, robot_odometry, robot_position, robot_speed, has_goal, jetson_vision_goal, timestamp):
    # ODOMETRY ENCODING
    odometry_x = robot_odometry[0]   
    odometry_y = robot_odometry[1]   
    odometry_theta = robot_odometry[2]

    # POSITION ENCODING
    position_x = robot_position[0]   
    position_y = robot_position[1]   
    position_theta = robot_position[2]

    # ODOMETRY ENCODING
    speed_x = robot_speed[0]   
    speed_y = robot_speed[1]   
    speed_w = robot_speed[2]

    # DETECTED BALL BOUNDING BOX
    xmin, xmax, ymin, ymax = jetson_vision_goal

    data = [frame_nr, id, odometry_x, odometry_y, odometry_theta, \
        position_x, position_y, position_theta, \
        speed_x, speed_y, speed_w, \
        has_goal, xmin, xmax, ymin, ymax, timestamp]
    return data

def get_bbox(vision, img):
    detections = vision.object_detector.inference(img).detections
    jetson_vision_goal = None
    for detection in detections:
        class_id, score, xmin, xmax, ymin, ymax = detection
        if class_id==2:
            jetson_vision_goal = [xmin, xmax, ymin, ymax]

    return jetson_vision_goal

if __name__ == "__main__":
    cwd = os.getcwd()

    # UDP COMMUNICATION SETUP
    eth_comm = communication_proto.SocketUDP()
   
    # VIDEO CAPTURE CONFIGS
    DISPLAY_WINDOW = False
    WINDOW_NAME = 'SELF LOCALIZATION DATASET'
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # INIT EMBEDDED VISION 
    vision = JetsonVision(draw=False, enable_field_detection=False, score_threshold=0.3)

    # FRAME NR COUNTER
    frame_nr = 1

    # DATA FOR LOGS
    QUADRADO_NR = 1
    ROBOT_ID = 0
    fields = ['FRAME_NR', 'ROBOT_ID', \
        'ODOMETRY X', 'ODOMETRY Y', 'ODOMETRY THETA', \
        'POSITION X', 'POSITION Y', 'POSITION THETA', \
        'SPEED X', 'SPEED Y', 'SPEED W', \
        'HAS GOAL', 'X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'TIMESTAMP']
    data_log = []
    start_time = time.time()
    save_frames = True   

    while True:
        # RECEIVE MSG FROM MCU
        ret_robot, robot_odometry, hasBall, kickLoad, battery, count, robot_position, robot_speed = eth_comm.recvSSLMessage()

        # SAVE TIMESTAMP BEFORE CAPTURING FRAME
        timestamp = time.time() - start_time

        # CAPTURE FRAME
        if cap.isOpened():
           ret_camera, frame = cap.read()
           if not ret_camera:
               print("Check video capture path")
               break
           else: img = frame

        # DETECT OBJECT'S BOUNDING BOXES
        jetson_vision_goal = get_bbox(vision, img)

        # CHECK IF ALL FIELDS ARE AVAILABLE
        has_goal = (jetson_vision_goal is not None)

        # UPDATE DATA AND PRINT FOR DEBUG
        if not has_goal:
            jetson_vision_goal = [0, 0, 0, 0]

        data = serialize_data_to_log(frame_nr, ROBOT_ID, robot_odometry, robot_position, robot_speed, has_goal, jetson_vision_goal, timestamp)

        # DISPLAY WINDOW FOR DEBUG
        if DISPLAY_WINDOW:
            cv2.moveWindow(WINDOW_NAME, 100, 50)
            cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        # ADD DETECTIONS TO LOG IF ROBOT HAS BALL ON SENSOR
        if save_frames:
            # SAVE RAW FRAME
            dir = cwd+f"/data/quadrado{QUADRADO_NR}/{frame_nr}.jpg"
            cv2.imwrite(dir, frame)

        # PRINT SAVED FRAME
        print(f"FRAME NR: {frame_nr} | ODOMETRY: {robot_odometry} | POSITION: {robot_position} | SPEED: {robot_speed} | HAS_GOAL: {int(has_goal)} | TIME: {timestamp}")

        # APPEND DATA TO LOG
        data_log.append(data)

        # ADD FRAME NR
        frame_nr += 1

        # FINISH AND SAVE LOG IF ROBOT IS TURNED OFF (OR RESET)
        if hasBall:
            print("FINISH CAPTURE!")
            break
        
    dir = cwd+f"/data/quadrado{QUADRADO_NR}/log.csv"
    with open(dir, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(data_log)

    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()
