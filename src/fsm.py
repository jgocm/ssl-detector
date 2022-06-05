from enum import Enum
from entities import Robot, Goal, Ball, Frame
from navigation import TargetPoint
import communication_proto

class State(Enum):
    unkown = 0
    stop = 1
    search = 2
    drive = 3
    dock = 4
    rotate = 5

class FSM():
    def __init__(
                self,
                initial_state = 1,
                init_time = 0
                ):
        super(FSM, self).__init__()
        self.current_state = State(initial_state)
        self.last_state = State(initial_state)
        self.state_init_time = init_time

    def getStateDuration(self, current_timestamp):
        return current_timestamp - self.state_init_time
    
    def stage1(self, frame = Frame(), ball = Ball(), robot = Robot()):
        """
        State Machine for Vision Blackout Stage 1:
        1) Initiates stopped for 0.5 seconds
        2) Robot rotates around its axis searching for the ball
        3) Sets ball as target point and drives to it
        4) After losing the ball, uses last seen position as target and navigates using inertial odometry only

        Inputs
        frame: current camera frame
        ball: last seen ssl ball
        robot: ssl robot
        --------------------
        Returns:
        target: ball position and direction
        """
        target = TargetPoint(x = 0, y = 0, w = 0)
    
        while True:
            final_state = self.current_state

            if self.current_state == State.stop:
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = State.search

            elif self.current_state == State.search:
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                if frame.has_ball:
                    final_state = State.drive

            elif self.current_state == State.drive:
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x - robot.camera_offset/1000,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = 0
                    )
                target.type = communication_proto.pb.protoPositionSSL.posType.target
                if not frame.has_ball and target.getDistance() < 0.270:
                    final_state = State.dock

            elif self.current_state == State.dock:
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x - robot.camera_offset/1000,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = 0
                    )
                target.type = communication_proto.pb.protoPositionSSL.posType.dock
                if frame.has_ball:
                    final_state = State.drive

            transition = (final_state != self.current_state)
            if transition == True:
                self.state_init_time = frame.timestamp
                self.last_state = self.current_state
                self.current_state = final_state
            else:
                break
        
        return target