import cv2
import os
import time
import communication_proto

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

    linha_nr = 1
    frame_nr = 1

    while True:
        odometry, hasBall, kickLoad, battery = eth_comm.recvSSLMessage()

        if battery>14:
            # CAPTURE FRAME
            _, frame = cap.read()

            # SAVE FRAME
            dir = cwd+f"/data/linha{linha_nr}/{frame_nr}.jpg"
            cv2.imwrite(dir, frame)

            # SAVE ODOMETRY DATA

            # ADD FRAME COUNTER
            frame_nr += 1
        
    cap.release()
    cv2.destroyAllWindows()
