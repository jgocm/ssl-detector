from enum import Enum
import math
from entities import Robot, Goal, Ball, Frame
from navigation import TargetPoint
import communication_proto
import numpy as np

class Stage(Enum):
    unknown = 0
    grabStationaryBall = 1
    scoreOnEmptyGoal = 2
    scoreOnDefendedGoal = 3
    moveToPoint = 4
    passAndScoreOnEmptyGoal = 5

class Stage1States(Enum):
    unkown = 0
    init = 1
    search = 2
    drive = 3
    dock = 4
    rotate = 5

class Stage2States(Enum):
    unkown = 0
    init = 1
    search = 2
    drive1 = 3
    stop1 = 4
    align1 = 5
    rotate = 6
    stop2 = 7
    drive2 = 8
    stop3 = 9
    align2 = 10
    drive3 = 11
    dock = 12

# STILL NOT IMPLEMENTED
class Stage3States(Enum):
    unkown = 0
    init = 1
    search = 2
    drive = 3
    dock = 4
    rotate = 5

# STILL NOT IMPLEMENTED
class Stage4States(Enum):
    unkown = 0
    init = 1
    search = 2
    drive = 3
    dock = 4
    rotate = 5

# STILL NOT IMPLEMENTED
class Stage5States(Enum):
    unkown = 0
    init = 1
    search = 2
    drive = 3
    dock = 4
    rotate = 5

class FSM():
    def __init__(
                self,
                stage = 1,
                initial_state = 1,
                init_time = 0
                ):
        super(FSM, self).__init__()
        self.stage = Stage(stage)
        if stage == 1:
            self.current_state = Stage1States(initial_state)
            self.last_state = Stage1States(initial_state)
        elif stage == 2:
            self.current_state = Stage2States(initial_state)
            self.last_state = Stage2States(initial_state)
        self.state_init_time = init_time

    def getStateDuration(self, current_timestamp):
        return current_timestamp - self.state_init_time
    
    def moveNStates(self, n):
        state_nr = self.current_state.value + n
        return state_nr
    
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

            if self.current_state == Stage1States.init:
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.search:
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.drive:
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = 0
                    )
                target.type = communication_proto.pb.protoPositionSSL.target
                if not frame.has_ball and target.getDistance() < 0.270:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.dock:
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -robot.camera_offset/1000
                    )
                target.type = communication_proto.pb.protoPositionSSL.dock
                if frame.has_ball:
                    final_state = self.moveNStates(-1)

            final_state = Stage1States(final_state)
            transition = (final_state != self.current_state)
            if transition == True:
                self.state_init_time = frame.timestamp
                self.last_state = self.current_state
                self.current_state = final_state
            else:
                break
        
        return target

    def stage2(self, frame = Frame(), ball = Ball(), goal = Goal(), robot = Robot()):
        target = TargetPoint(x = 0, y = 0, w = 0)
    
        while True:
            final_state = self.current_state

            if self.current_state.value == 1:
                # init state: waits for 0.5 seconds
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)

            elif self.current_state.value == 2:
                # search: rotate on self searching for ball
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state.value == 3:
                # drive1: goes in ball direction preparing to rotate in point
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5-robot.camera_offset/1000
                    )
                target.type = communication_proto.pb.protoPositionSSL.target
                if target.getDistance() < 0.05:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 4:
                # stop1: breaks for align with ball center
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 5:
                # align1: adjust rotation to align with ball center
                target.type = communication_proto.pb.protoPositionSSL.rotateControl
                target.setPosition(ball.x, ball.y)
                target.w = target.getDirection()
                if np.abs(target.w) < 0.125:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 6:
                # rotate: rotates around the ball searching for a goal
                target.type = communication_proto.pb.protoPositionSSL.rotateInPoint
                if frame.has_ball and frame.has_goal:
                    _, _, w = target.get2XYCoordinatesVector(
                        x1 = ball.x,
                        y1 = ball.y,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                    if np.abs(w) < 0.3:
                        final_state = self.moveNStates(1)
            
            elif self.current_state.value == 7:
                # stop2: breaks when ball and goal are aligned
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 8:
                # drive2: drive to nearer point aligning with the goal->ball line
                target.type = communication_proto.pb.protoPositionSSL.target
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = goal.center_x,
                    y1 = goal.center_y,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = 0.3
                    )
                target.w = math.pi-target.w
                if np.abs(target.w)<0.05 and target.getDistance()<0.05:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 9:
                # stop3: breaks after moving to nearer point
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)

            elif self.current_state.value == 10:
                # align2: adjusts orientation to goal->ball line
                target.type = communication_proto.pb.protoPositionSSL.rotateControl
                _, _, target.w = target.get2XYCoordinatesVector(
                    x1 = ball.x,
                    y1 = ball.y,
                    x2 = goal.center_x,
                    y2 = goal.center_y
                    )
                if np.abs(target)<0.125:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 11:
                # drive3: drives to ball
                target.type = communication_proto.pb.protoPositionSSL.target
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = 0
                    )
                if not frame.has_ball and target.getDistance()<0.270:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 12:
                # dock: drives to last target ball using inertial odometry and shoots to goal
                target.type = communication_proto.pb.protoPositionSSL.dock
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = 0
                    )
                robot.front = True
                robot.charge = False
                robot.kick_strength = 40
                if self.getStateDuration(frame.timestamp)>3:
                    robot.charge = True
                if self.getStateDuration(frame.timestamp)>3.1:
                    robot.charge = False

            final_state = Stage2States(final_state)
            transition = (final_state != self.current_state)
            if transition == True:
                self.state_init_time = frame.timestamp
                self.last_state = self.current_state
                self.current_state = final_state
            else:
                break
        
        return target, robot