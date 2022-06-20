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
    moveToPoint = 3
    passAndScoreOnEmptyGoal = 4

class Stage1States(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToBall = 3
    dockBall = 4

class Stage2States(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToRotationPoint = 3
    stopInRotationPoint = 4
    alignToRotationTarget = 5
    rotateInPoint = 6
    stopToShoot = 7
    driveToBall = 8
    dockAndShoot = 9
    stopToEnd = 10

# STILL NOT IMPLEMENTED
class Stage3States(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToBall = 3
    dockBall = 4

# STILL NOT IMPLEMENTED
class Stage4States(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToBall = 3
    dockBall = 4

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
                if self.getStateDuration(frame.timestamp) > 0.3:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.searchBall:
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.driveToBall:
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -robot.camera_offset/1000
                    )
                target.type = communication_proto.pb.protoPositionSSL.target
                if not frame.has_ball and target.getDistance() < 0.270:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.dockBall:
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
                # init state: waits for 0.3 seconds
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.3:
                    final_state = self.moveNStates(1)

            elif self.current_state.value == 2:
                # search: rotate on self searching for ball
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state.value == 3:
                # drive1: goes in ball direction preparing to rotate in point
                target.type = communication_proto.pb.protoPositionSSL.target
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5-robot.camera_offset/1000
                    )
                if target.getDistance() < 0.05:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 4:
                # stop1: breaks for align with ball center
                target.type = communication_proto.pb.protoPositionSSL.stop
                target = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5-robot.camera_offset/1000
                    )
                if target.getDistance() > 0.05:
                    final_state = self.moveNStates(-1)
                elif self.getStateDuration(frame.timestamp) > 0.5:
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
                target.setPosition(x=ball.x, y=ball.y)
                if frame.has_ball and frame.has_goal:
                    _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = ball.x,
                        y1 = ball.y,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                    if np.abs(target.w) < 0.03:
                        final_state = self.moveNStates(1)
                else:
                    target.w = math.pi

            elif self.current_state.value == 7:
                # stop2: breaks when ball and goal are aligned
                target.type = communication_proto.pb.protoPositionSSL.stop
                _, _, w = target.get2XYCoordinatesVector(
                        x1 = ball.x,
                        y1 = ball.y,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                if np.abs(w) > 0.035:
                    final_state = self.moveNStates(-1)
                elif self.getStateDuration(frame.timestamp) > 0.2:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 8:
                # drive3: drives to ball position towards goal direction
                target.type = communication_proto.pb.protoPositionSSL.target
                target.setPosition(x = ball.x-robot.camera_offset/1000, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                if not frame.has_ball and target.getDistance()<0.270:
                    final_state = self.moveNStates(1)
            
            elif self.current_state.value == 9:
                # dock: drives to last target ball using inertial odometry and shoots to goal
                target.type = communication_proto.pb.protoPositionSSL.dock
                target.setPosition(x = ball.x-robot.camera_offset/1000, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = goal.center_x,
                        y2 = goal.center_y
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