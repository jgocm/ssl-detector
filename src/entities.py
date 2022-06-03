import numpy as np
import math
import CommTypes_pb2 as pb

class Robot():
    def __init__(
                self,
                id = 0,
                height = 155,
                diameter = 180,
                camera_offset = 90
                ):
        super(Robot, self).__init__()
        self.id = id
        self.height = height
        self.diameter = diameter

        self.x = 0
        self.y = 0
        self.w = 0
        self.position = [self.x, self.y, self.w]
        self.rotation = None

        self.pose_confidence = 0
        self.camera_offset = camera_offset

        self.front = False
        self.chip = False
        self.charge = False
        self.kick_strength = 0

        self.dribbler = False
        self.dribbler_speed = 0
    
    def isLocated(self):
        if self.pose_confidence > 0.7:
            return True
        else:
            return False

    def getPose(self):
        if self.isLocated():
            print(f"Pose confidence is {self.pose_confidence}")
            return self.position, self.pose_confidence
        else:
            print(f"Robot pose is not known")
            return self.pose_confidence

    def getId(self):
        return self.id

    def updatePoseConfidence(self, confidence):
        self.pose_confidence = confidence
    
    def setPose(self, position, euler_angles):
        self.position = position
        self.rotation = euler_angles[2][0]
        self.updatePoseConfidence(confidence=1)
    
    def cameraToRobotCoordinates(self, x, y):
        robot_x = (y + self.camera_offset)/1000
        robot_y = -x/1000
        robot_w = math.atan2(robot_y, robot_x)

        return robot_x, robot_y, robot_w

class Ball():
    def __init__(
                self,
                diameter = 42.7,
                radius = 42.7/2
                ):
        super(Ball, self).__init__()
        self.x = 0
        self.y = 0
        self.w = 0
        self.radius = diameter/2

    def updatePosition(self, x, y):
        self.x, self.y, self.w = x, y, math.atan2(y, x)
    
    def getDistance(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def getDirection(self):
        self.w = math.atan2(self.y, self.x)
        return self.w

class Goal():
    def __init__(
                self,
                width = 1000,
                depth = 180,
                height = 155
                ):
        super(Goal, self).__init__()
        self.width = width
        self.depth = depth
        self.height = height
        self.center_x = 0
        self.center_y = 0

class Field():
    def __init__(
                self,
                field_width = 3760,
                field_length = 5640,
                penalty_area_width = 2000,
                penalty_area_depth = 1000,
                center_radius = 1000,
                boundary_width = 180,
                line_thickness = 20
                ):
        super(Field, self).__init__()
        self.width = field_width
        self.length = field_length
        self.penalty_area_width = penalty_area_width
        self.penalty_area_depth = penalty_area_depth
        self.boundary_width = boundary_width
        self.line_thickness = line_thickness
        self.center_radius = center_radius
        self.goal = Goal()
    
    def getGoalCoordinates(self):
        p1 = -self.goal.width/2, self.length/2
        p2 = self.goal.width/2, self.length/2
        return p1, p2
    
class Frame():
    def __init__(self):
        super(Frame, self).__init__()
        self.input = None
        self.ball = Ball()
        self.goal = Goal()
        self.robot = Robot()
        self.has_ball = False
        self.has_goal = False
        self.has_robot = False
        self.has_target = False
    
    def updateBall(self, x, y):
        self.ball.updatePosition(x, y)
        self.has_ball = True
        return self.ball

    def updateGoalCenter(self, x, y):
        self.goal.center_x = x
        self.goal.center_y = y
        self.has_goal = True
        return self.goal

    def updateRobot(self, x, y):
        self.robot.x = x
        self.robot.y = y
        self.has_robot = True
        return self.robot

class GroundPoint():
    def __init__(
                self,
                x0 = 0,
                y0 = 0
                ):
        super(GroundPoint, self).__init__()
        self.x = x0
        self.y = y0
        self.type = pb.protoPositionSSL.unknown
    
    def setPosition(self, x, y):
        self.x = x
        self.y = y

    def getDistance(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def getDirection(self):
        self.w = math.atan2(self.y, self.x)
        return self.w

    def offsetTarget(self, x, y, w, offset=1):
        dist = math.sqrt(x**2+y**2) + 0.001
        prop = (dist - offset)/dist
        target_x, target_y, target_w = prop*x, prop*y, w
        return target_x, target_y, target_w

    def directionVector(self, x1, y1, x2, y2):
        vy = y2-y1
        vx = x2-x1
        w = math.atan2(-vy, -vx)
        return -vx, -vy

    def alignedTarget(self, vx, vy, bx, by, offset):
        v_norm = math.sqrt(vx**2+vy**2)
        target_x = vx*offset/v_norm + bx
        target_y = vy*offset/v_norm + by
        target_w = math.atan2(target_y, target_x)
        return target_x, target_y, target_w