import cv2
import time
import os

# LOCAL IMPORTS
import communication_proto
from fsm import FSM, Stage1States
from jetson_vision import JetsonVision

def main():
    cwd = os.getcwd()

    # START TIME
    start = time.time()
    EXECUTION_TIME = 60

    # UDP COMMUNICATION SETUP
    eth_comm = communication_proto.SocketUDP()

    # VIDEO CAPTURE CONFIGS
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # EMBEDDED VISION SETUP
    vision = JetsonVision(vertical_lines_offset=320)

    # INIT VISION BLACKOUT STATE MACHINE
    INITIAL_STATE = 1
    STAGE = 1
    state_machine = FSM(
        stage = STAGE,
        initial_state = INITIAL_STATE,
        init_time = start)

    # CONFIGURE AND LOAD VISION SOURCE
    _, init_src = cap.read()
    vision.process(init_src, 0)
    config_time = time.time() - start
    print(f"Configuration Time: {config_time:.2f}s")

    while cap.isOpened():

        # CAPTURE FRAME
        timestamp = time.time()-config_time
        ret, frame = cap.read()
        if not ret:
            print("Check video capture path")
            break
        
        # PROCESS VISION
        processed_vision = vision.process(frame, timestamp)
        current_frame, tracked_ball, tracked_goal, tracked_robot, particle_filter_observations = processed_vision

        # COMPUTE SELF-LOCALIZATION
        odometry = eth_comm.recvSSLMessage()

        # STATE MACHINE
        target, ssl_robot = state_machine.stage1(
                                frame = current_frame, 
                                ball = tracked_ball,
                                goal = tracked_goal, 
                                robot = tracked_robot)
    
        # UPDATE PROTO MESSAGE
        eth_comm.setSSLMessage(target = target, robot = ssl_robot)
        
        # ACTION
        eth_comm.sendSSLMessage()
        print(f'{state_machine.current_state} | odometry: {odometry[0]} | reset odometry: {eth_comm.msg.resetOdometry}')
        # print(f'{state_machine.current_state} \
        #     | {state_machine.getStateDuration(current_timestamp=current_frame.timestamp)} \
        #     | Target: {eth_comm.msg.x:.3f}, \
        #         {eth_comm.msg.y:.3f}, \
        #         {eth_comm.msg.w:.3f}, \
        #         {eth_comm.msg.PosType.Name(eth_comm.msg.posType)} \
        #     | speed: {eth_comm.msg.max_speed}, {eth_comm.msg.min_speed}')

        if state_machine.current_state == Stage1States.finish and \
            state_machine.getStateDuration(current_timestamp=current_frame.timestamp)>1:
            break
            
        # CHECK FOR DURATION TIMEOUT
        if time.time()-config_time-start>EXECUTION_TIME:
            eth_comm.sendStopMotion()
            break

    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()