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
    passToAlly = 4
    receiveAndScoreOnEmptyGoal = 5

class Stage1States(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToBall = 3
    dockBall = 4
    finish = 5

class Stage2States(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToRotationPoint = 3
    stopInRotationPoint = 4
    alignToRotationTarget = 5
    rotateSearchGoal = 6
    rotateAlignToGoal = 7
    stopToShoot = 8
    driveToBall = 9
    driveTowardsGoal = 10
    dockAndShoot = 11
    finish = 12

# STILL NOT IMPLEMENTED
class Stage3States(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToBall = 3
    dockBall = 4

class Stage4PasserStates(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    driveToRotationPoint = 3
    stopInRotationPoint = 4
    alignToRotationTarget = 5
    rotateSearchRobot = 6
    rotateAlignToRobot = 7
    stopToPass = 8
    driveToBall = 9
    driveTowardsAlly = 10
    dockAndPass = 11
    moveBackwards = 12
    finish = 13

class Stage4ReceiverStates(Enum):
    unkown = 0
    init = 1
    searchBall = 2
    alignToBall = 3
    waitAllyPositioning = 4
    driveToReceivingPoint = 5
    stopInReceivingPoint = 6
    alignToReceive = 7
    waitPass = 8
    moveBackwards = 9
    relocalizeBall = 10
    driveToRotationPoint = 11
    stopInRotationPoint = 12
    alignToRotationTarget = 13
    rotateSearchGoal = 14
    rotateAlignToGoal = 15
    stopToShoot = 16
    driveToBall = 17
    driveTowardsGoal = 18
    dockAndShoot = 19
    finish = 20

class FSM():
    def __init__(
                self,
                stage = 1,
                initial_state = 1,
                init_time = 0
                ):
        super(FSM, self).__init__()
        self.stage = Stage(stage)
        self.target = TargetPoint()
        self.ssl_robot = Robot()
        
        if stage == 1:
            self.current_state = Stage1States(initial_state)
            self.last_state = Stage1States(initial_state)
        elif stage == 2:
            self.current_state = Stage2States(initial_state)
            self.last_state = Stage2States(initial_state)
        elif stage == 4:
            self.current_state = Stage4PasserStates(initial_state)
            self.last_state = Stage4PasserStates(initial_state)
        elif stage == 5:
            self.current_state = Stage4ReceiverStates(initial_state)
            self.last_state = Stage4ReceiverStates(initial_state)
        self.state_init_time = init_time

    def getStateDuration(self, current_timestamp):
        return current_timestamp - self.state_init_time
    
    def moveNStates(self, n):
        state_nr = self.current_state.value + n
        return state_nr
    
    def stage1(self, frame = Frame(), ball = Ball(), goal = Goal(), robot = Robot()):
        """
        State Machine for Vision Blackout Stage 1:
        1) Initiates stopped for 0.1 seconds
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
        robot: ssl robot commands
        """
        target = TargetPoint(x = 0, y = 0, w = 0)
    
        while True:
            final_state = self.current_state

            if self.current_state == Stage1States.init:
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.1:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.searchBall:
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.w = math.pi
                target.max_speed = 1.5
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.driveToBall:
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = 0
                    )
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                if not frame.has_ball and target.getDistance() < 0.400:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage1States.dockBall:
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.12
                    )
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.reset_odometry = False
                if self.getStateDuration(current_timestamp=frame.timestamp) > 3:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage1States.finish:
                target.type = communication_proto.pb.protoPositionSSL.stop

            final_state = Stage1States(final_state)
            transition = (final_state != self.current_state)
            if transition == True:
                self.state_init_time = frame.timestamp
                self.last_state = self.current_state
                self.current_state = final_state
            else:
                break
        
        return target, robot

    def stage2(self, frame = Frame(), ball = Ball(), goal = Goal(), robot = Robot()):
        target = TargetPoint(x = 0, y = 0, w = 0)

        while True:
            final_state = self.current_state

            if self.current_state == Stage2States.init:
                # init state: waits for 0.3 seconds
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.2:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage2States.searchBall:
                # search: rotate on self searching for ball
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.w = math.pi
                target.max_speed = 1.5
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage2States.driveToRotationPoint:
                # drive1: goes in ball direction preparing to rotate in point
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                robot.charge = True
                if target.getDistance() < 0.05:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage2States.stopInRotationPoint:
                # stop1: breaks for align with ball center
                target.type = communication_proto.pb.protoPositionSSL.stop
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                if target.getDistance() > 0.05:
                    final_state = self.moveNStates(-1)
                elif self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage2States.alignToRotationTarget:
                # align1: adjust rotation to align with ball center
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.setPosition(ball.x, ball.y)
                target.w = target.getDirection()
                if np.abs(target.w) < 0.125:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage2States.rotateSearchGoal:
                # rotate: rotates around the ball searching for a goal
                target.type = communication_proto.pb.protoPositionSSL.rotateInPoint
                target.setPosition(x=ball.x, y=ball.y)

                if goal.center_x > 0:
                    target.w = math.pi/6
                else:
                    target.w = math.pi/6

                if goal.center_y < 0:
                    target.w = -target.w
                if frame.has_ball and frame.has_goal:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage2States.rotateAlignToGoal:
                # rotate: rotates around the ball searching for a goal
                target.type = communication_proto.pb.protoPositionSSL.rotateInPoint
                target.setPosition(x=ball.x, y=ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                    x1 = ball.x,
                    y1 = ball.y,
                    x2 = goal.center_x,
                    y2 = goal.center_y
                    )
                if frame.has_goal:
                    target.reset_odometry = True
                else:
                    target.reset_odometry = False

                if (np.abs(target.w) < 0.025) or (self.getStateDuration(frame.timestamp) > 5):
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage2States.stopToShoot:
                # stop2: breaks when ball and goal are aligned
                target.type = communication_proto.pb.protoPositionSSL.stop
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = ball.x,
                        y1 = ball.y,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                if frame.has_goal:
                    final_state = self.moveNStates(2)
                elif self.getStateDuration(frame.timestamp) > 0.2:
                    final_state = self.moveNStates(1)

            
            elif self.current_state == Stage2States.driveToBall:
                # drive3: drives to ball position towards goal direction
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = ball.x,
                        y2 = ball.y
                        )
                if not frame.has_ball and target.getDistance()<0.400:
                    final_state = self.moveNStates(2)

            elif self.current_state == Stage2States.driveTowardsGoal:
                # drive3: drives to ball position towards goal direction
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                if not frame.has_ball and target.getDistance()<0.400:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage2States.dockAndShoot:
                # dock: drives to last target ball using inertial odometry and shoots to goal
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                if self.last_state == Stage2States.driveTowardsGoal:
                    _, _, target.w = target.get2XYCoordinatesVector(
                            x1 = 0,
                            y1 = 0,
                            x2 = goal.center_x,
                            y2 = goal.center_y
                            )
                elif self.last_state == Stage2States.driveToBall:
                    _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = ball.x,
                        y2 = ball.y
                        )
                target.reset_odometry = False
<<<<<<< HEAD
                robot.front = True
                robot.charge = False
                robot.kick_strength = 40
                if self.getStateDuration(frame.timestamp) > 3:
                    robot.front = False
                    final_state = self.moveNStates(1)
              
=======
                robot.charge = False
                robot.front = True
                robot.kick_strength = 60
                if self.getStateDuration(frame.timestamp) > 3:
                    robot.front = False
                    robot.kick_strength = 0
                    final_state = self.moveNStates(1)
                
>>>>>>> origin/particle_filter_development
            elif self.current_state == Stage2States.finish:
                target.type = communication_proto.pb.protoPositionSSL.stop

            final_state = Stage2States(final_state)
            transition = (final_state != self.current_state)
            if transition == True:
                self.state_init_time = frame.timestamp
                self.last_state = self.current_state
                self.current_state = final_state
            else:
                break
        
        return target, robot

    def stage4Passer(self, frame = Frame(), ball = Ball(), ally = Robot(), robot = Robot()):
        target = TargetPoint(x = 0, y = 0, w = 0)

        while True:
            final_state = self.current_state

            if self.current_state == Stage4PasserStates.init:
                # init state: waits for 0.3 seconds
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.2:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4PasserStates.searchBall:
                # search: rotate on self searching for ball
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.w = math.pi
                target.max_speed = 1.5
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4PasserStates.driveToRotationPoint:
                # drive1: goes in ball direction preparing to rotate in point
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                if target.getDistance() < 0.05:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4PasserStates.stopInRotationPoint:
                # stop1: breaks for align with ball center
                target.type = communication_proto.pb.protoPositionSSL.stop
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                if target.getDistance() > 0.05:
                    final_state = self.moveNStates(-1)
                elif self.getStateDuration(frame.timestamp) > 0.2:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4PasserStates.alignToRotationTarget:
                # align1: adjust rotation to align with ball center
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.max_speed = 0.5
                target.setPosition(ball.x, ball.y)
                target.w = target.getDirection()
                if np.abs(target.w) < 0.125:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4PasserStates.rotateSearchRobot:
                # rotate: rotates around the ball searching for a goal
                target.type = communication_proto.pb.protoPositionSSL.rotateInPoint
                target.setPosition(x=ball.x, y=ball.y)

                if ally.x > 0:
                    target.w = math.pi/6
                else:
                    target.w = math.pi/6

                if ally.y < 0:
                    target.w = -target.w
                if frame.has_ball and frame.has_robot:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4PasserStates.rotateAlignToRobot:
                # rotate: rotates around the ball searching for a goal
                target.type = communication_proto.pb.protoPositionSSL.rotateInPoint
                target.setPosition(x=ball.x, y=ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                    x1 = ball.x,
                    y1 = ball.y,
                    x2 = ally.x,
                    y2 = ally.y
                    )
                if frame.has_robot:
                    target.reset_odometry = True
                else:
                    target.reset_odometry = False

                dist = math.sqrt((ball.x - ally.x)**2 + (ball.y - ally.y)**2)
                if (np.abs(target.w) < 0.025) and (dist < 4):
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4PasserStates.stopToPass:
                # stop2: breaks when ball and goal are aligned
                target.type = communication_proto.pb.protoPositionSSL.stop
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = ball.x,
                        y1 = ball.y,
                        x2 = ally.x,
                        y2 = ally.y
                        )
                robot.charge = True

                if frame.has_robot:
                    final_state = self.moveNStates(2)
                elif self.getStateDuration(frame.timestamp) > 0.3:
                    final_state = self.moveNStates(1)

            
            elif self.current_state == Stage4PasserStates.driveToBall:
                # drive3: drives to ball position towards goal direction
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = ball.x,
                        y2 = ball.y
                        )
                robot.charge = True
                if not frame.has_ball and target.getDistance()<0.400:
                    final_state = self.moveNStates(2)

            elif self.current_state == Stage4PasserStates.driveTowardsAlly:
                # drive3: drives to ball position towards goal direction
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = ally.x,
                        y2 = ally.y
                        )
                robot.charge = True
                if not frame.has_ball and target.getDistance()<0.400:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4PasserStates.dockAndPass:
                # dock: drives to last target ball using inertial odometry and shoots to goal
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                if self.last_state == Stage4PasserStates.driveTowardsAlly:
                    _, _, target.w = target.get2XYCoordinatesVector(
                            x1 = 0,
                            y1 = 0,
                            x2 = ally.x,
                            y2 = ally.y
                            )
                elif self.last_state == Stage4PasserStates.driveToBall:
                    _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = ball.x,
                        y2 = ball.y
                        )
                dist = math.sqrt(ally.x**2 + ally.y**2)
                target.reset_odometry = False
                robot.charge = False
                robot.front = True
                robot.kick_strength = 6*dist + 8.4
                if dist>3:
                    robot.kick_strength = 30
                if self.getStateDuration(frame.timestamp) > 3:
                    robot.front = False
                    robot.kick_strength = 0
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4PasserStates.moveBackwards:
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.x = -1
                target.reset_odometry = False

                if self.getStateDuration(frame.timestamp) > 1:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4PasserStates.finish:
                target.type = communication_proto.pb.protoPositionSSL.stop

            final_state = Stage4PasserStates(final_state)
            transition = (final_state != self.current_state)
            if transition == True:
                self.state_init_time = frame.timestamp
                self.last_state = self.current_state
                self.current_state = final_state
            else:
                break
                
        return target, robot
    
    def stage4Receiver(self, frame = Frame(), ball = Ball(), goal = Goal(), ally = Robot(), robot = Robot()):
        target = TargetPoint(x = 0, y = 0, w = 0)
        
        while True:
            final_state = self.current_state
        
            if self.current_state == Stage4ReceiverStates.init:
                # init state: waits for 0.3 seconds
                target.type = communication_proto.pb.protoPositionSSL.stop
                if self.getStateDuration(frame.timestamp) > 0.2:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4ReceiverStates.searchBall:
                # search: rotate on self searching for ball
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                if frame.has_ball:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.alignToBall:
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.setPosition(ball.x, ball.y)
                target.w = target.getDirection()
                if np.abs(target.w) < 0.125 and frame.has_robot:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.waitAllyPositioning:
                target.type = communication_proto.pb.protoPositionSSL.stop
                target.setPosition(x=ball.x, y=ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                    x1 = ball.x,
                    y1 = ball.y,
                    x2 = ally.x,
                    y2 = ally.y
                    )

                if frame.has_ball and frame.has_robot and np.abs(target.w) < 0.125:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.driveToReceivingPoint:
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                if target.getDistance() < 0.05:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4ReceiverStates.stopInReceivingPoint:
                # stop1: breaks for align with ball center
                target.type = communication_proto.pb.protoPositionSSL.stop
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                if target.getDistance() > 0.05:
                    final_state = self.moveNStates(-1)
                elif self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)           

            elif self.current_state == Stage4ReceiverStates.alignToReceive:
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.setPosition(ball.x, ball.y)
                target.w = target.getDirection()
                if np.abs(target.w) < 0.05 and frame.has_robot:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.waitPass:
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.setPosition(ball.x, ball.y)
                target.w = target.getDirection()

                if not frame.has_ball and target.getDistance()<0.500:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.moveBackwards:
                if self.getStateDuration(frame.timestamp)>2:
                    target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                    target.x = -1
                    target.reset_odometry = False
                else:
                    target.type = communication_proto.pb.protoPositionSSL.stop
   
                if self.getStateDuration(frame.timestamp) > 3:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.relocalizeBall:
                # search: rotate on self searching for ball
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                if frame.has_ball:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4ReceiverStates.driveToRotationPoint:
                # drive1: goes in ball direction preparing to rotate in point
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                if target.getDistance() < 0.05:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.stopInRotationPoint:
                # stop1: breaks for align with ball center
                target.type = communication_proto.pb.protoPositionSSL.stop
                target.x, target.y, target.w = target.getTargetRelativeToLine2DCoordinates(
                    x1 = 0,
                    y1 = 0,
                    x2 = ball.x,
                    y2 = ball.y,
                    relative_angle = 0,
                    relative_distance = -0.5
                    )
                if target.getDistance() > 0.05:
                    final_state = self.moveNStates(-1)
                elif self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.alignToRotationTarget:
                # align1: adjust rotation to align with ball center
                target.type = communication_proto.pb.protoPositionSSL.rotateOnSelf
                target.setPosition(ball.x, ball.y)
                target.w = target.getDirection()
                if np.abs(target.w) < 0.125:
                    final_state = self.moveNStates(1)
            
            elif self.current_state == Stage4ReceiverStates.rotateSearchGoal:
                # rotate: rotates around the ball searching for a goal
                target.type = communication_proto.pb.protoPositionSSL.rotateInPoint
                target.setPosition(x=ball.x, y=ball.y)

                if goal.center_x > 0:
                    target.w = math.pi/6
                else:
                    target.w = math.pi/2

                if goal.center_y < 0:
                    target.w = -target.w
                if frame.has_ball and frame.has_goal:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4ReceiverStates.rotateAlignToGoal:
                # rotate: rotates around the ball searching for a goal
                target.type = communication_proto.pb.protoPositionSSL.rotateInPoint
                target.setPosition(x=ball.x, y=ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                    x1 = ball.x,
                    y1 = ball.y,
                    x2 = goal.center_x,
                    y2 = goal.center_y
                    )
                if frame.has_goal:
                    target.reset_odometry = True
                else:
                    target.reset_odometry = False

                if (np.abs(target.w) < 0.025) or (self.getStateDuration(frame.timestamp) > 5):
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4ReceiverStates.stopToShoot:
                # stop2: breaks when ball and goal are aligned
                target.type = communication_proto.pb.protoPositionSSL.stop
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = ball.x,
                        y1 = ball.y,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                robot.charge = True

                if frame.has_goal:
                    final_state = self.moveNStates(2)
                elif self.getStateDuration(frame.timestamp) > 0.5:
                    final_state = self.moveNStates(1)

            
            elif self.current_state == Stage4ReceiverStates.driveToBall:
                # drive3: drives to ball position towards goal direction
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = ball.x,
                        y2 = ball.y
                        )
                robot.charge = True
                if not frame.has_ball and target.getDistance()<0.400:
                    final_state = self.moveNStates(2)

            elif self.current_state == Stage4ReceiverStates.driveTowardsGoal:
                # drive3: drives to ball position towards goal direction
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = goal.center_x,
                        y2 = goal.center_y
                        )
                robot.charge = True
                if not frame.has_ball and target.getDistance()<0.400:
                    final_state = self.moveNStates(1)

            elif self.current_state == Stage4ReceiverStates.dockAndShoot:
                # dock: drives to last target ball using inertial odometry and shoots to goal
                target.type = communication_proto.pb.protoPositionSSL.driveToTarget
                target.setPosition(x = ball.x, y = ball.y)
                if self.last_state == Stage4ReceiverStates.driveTowardsGoal:
                    _, _, target.w = target.get2XYCoordinatesVector(
                            x1 = 0,
                            y1 = 0,
                            x2 = goal.center_x,
                            y2 = goal.center_y
                            )
                elif self.last_state == Stage4ReceiverStates.driveToBall:
                    _, _, target.w = target.get2XYCoordinatesVector(
                        x1 = 0,
                        y1 = 0,
                        x2 = ball.x,
                        y2 = ball.y
                        )
                target.reset_odometry = False
                robot.charge = True
                if robot.front:
                    robot.front = False
                    robot.charge = False
                    final_state = self.moveNStates(1)
                elif self.getStateDuration(frame.timestamp) > 3:
                    robot.front = True
                    robot.kick_strength = 40
                
            elif self.current_state == Stage4ReceiverStates.finish:
                target.type = communication_proto.pb.protoPositionSSL.stop

            final_state = Stage4ReceiverStates(final_state)
            transition = (final_state != self.current_state)
            if transition == True:
                self.state_init_time = frame.timestamp
                self.last_state = self.current_state
                self.current_state = final_state
            else:
                break
        
        return target, robot   




