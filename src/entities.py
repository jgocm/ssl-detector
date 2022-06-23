import numpy as np
import math

class Robot():
    def __init__(
                self,
                id = 0,
                height = 155,
                diameter = 180,
                camera_offset = 90,
                initial_pose = [0, 0, 0]
                ):
        self.id = id
        self.height = height
        self.diameter = diameter
        self.score = 0

        self.x = 0
        self.y = 0
        
        self.tx = initial_pose[0]
        self.ty = initial_pose[1]
        self.w = initial_pose[2]
        self.is_located = False

        self.camera_offset = camera_offset

        self.front = False
        self.chip = False
        self.charge = False
        self.kick_strength = 0
        self.dribbler = False
        self.dribbler_speed = 0
    
    def updateSelfPose(self, x, y, w):
        self.tx = x
        self.ty = y
        self.w = w
        self.is_located = True

    def cameraToRobotCoordinates(self, x, y):
        robot_x = (y + self.camera_offset)/1000
        robot_y = -x/1000
        robot_w = math.atan2(robot_y, robot_x)

        return robot_x, robot_y, robot_w
    
    def cameraToRobotRotation(self, w):
        robot_w = w - math.pi/2

        return robot_w

class Ball():
    def __init__(
                self,
                diameter = 42.7,
                radius = 42.7/2
                ):
        self.score = 0
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
        self.width = width
        self.depth = depth
        self.height = height
        self.center_x = 0
        self.center_y = 0
        self.left_x = 0
        self.left_y = 0
        self.right_x = 0
        self.right_y = 0
        self.score = 0

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
        self.width = field_width
        self.length = field_length
        self.penalty_area_width = penalty_area_width
        self.penalty_area_depth = penalty_area_depth
        self.boundary_width = boundary_width
        self.line_thickness = line_thickness
        self.center_radius = center_radius
        self.goal = Goal()
        self.goal.center_x, self.goal.center_y = self.length/2, 0
    
class Frame():
    def __init__(
                self,
                timestamp = 0):
        self.input = None
        self.ball = Ball()
        self.goal = Goal()
        self.robot = Robot()
        self.has_ball = False
        self.has_goal = False
        self.has_robot = False
        self.has_target = False
        self.timestamp = timestamp
    
    def updateBall(self, x, y, score):
        if score >= self.ball.score:
            self.ball.updatePosition(x, y)
            self.has_ball = True
            self.ball.score = score
        return self.ball

    def updateGoalCenter(self, x, y, score):
        if score >= self.goal.score: 
            self.goal.center_x = x
            self.goal.center_y = y
            self.has_goal = True
            self.goal.score = score
        return self.goal

    def updateGoalCorners(self, left_corner_x, left_corner_y, right_corner_x, right_corner_y, score):
        if score >= self.goal.score:
            self.goal.left_x = left_corner_x
            self.goal.left_y = left_corner_y
            self.goal.right_x = right_corner_x
            self.goal.right_y = right_corner_y
            self.has_goal = True
            self.goal.score = score
        return self.goal

    def updateRobot(self, x, y, score):
        if score >= self.robot.score:
            self.robot.x = x
            self.robot.y = y
            self.has_robot = True
            self.robot.score = score
        return self.robot