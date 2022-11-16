import cv2
import os
import time
import numpy as np
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

    quadrado_nr = 2
    frame_nr = 1

    while True:
        odometry, hasBall, kickLoad, battery, count = eth_comm.recvSSLMessage()

        if battery>14:
            # CAPTURE FRAME
            _, frame = cap.read()

            # SAVE FRAME
            dir = cwd+f"/data/quadrado{quadrado_nr}/{frame_nr}_{count}.jpg"
            cv2.imwrite(dir, frame)

            # SAVE ODOMETRY DATA
            np.savetxt(cwd+f"/data/quadrado{quadrado_nr}/{frame_nr}_{count}.txt", odometry)
            
            # ADD FRAME NR
            frame_nr += 1
            
            # PRINT FOR DEBUG
            print(f"odometry: {odometry} | count: {count:.0f}")

    cap.release()
    cv2.destroyAllWindows()
