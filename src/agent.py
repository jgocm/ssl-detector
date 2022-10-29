import numpy as np
import math

class Agent():
    def __init__(
                self,
                id = 0,
                height = 155,
                diameter = 180,
                camera_offset = 90,
                initial_pose = [0, 0, 0]
                ):

        # fixed properties
        self.id = id
        self.height = height                # robot height in mm
        self.diameter = diameter            # robot diameter in mm
        self.camera_offset = camera_offset  # robot center to camera distance in mm
        
        # agent localization estimated from particles filter
        self.x = initial_pose[0]
        self.y = initial_pose[1]
        self.w = initial_pose[2]
        self.pose_confidence = 0
        self.is_located = False

        # information from low-level sensing
        self.movement = [0, 0, 0]
        self.has_ball = False
        self.kick_load = 0
        self.battery = 0

        # kick and dribbler commands
        self.front = False
        self.chip = False
        self.charge = False
        self.kick_strength = 0
        self.dribbler = False
        self.dribbler_speed = 0
    
    def updateSelfPose(self, x, y, w, confidence = 1):
        self.x = x
        self.y = y
        self.w = w
        self.is_located = True
        self.pose_confidence = confidence

    def cameraToRobotCoordinates(self, x, y):
        robot_x = (y + self.camera_offset)/1000
        robot_y = -x/1000
        robot_w = math.atan2(robot_y, robot_x)

        return robot_x, robot_y, robot_w
    
    def cameraToRobotRotation(self, w):
        robot_w = w - math.pi/2

        return robot_w