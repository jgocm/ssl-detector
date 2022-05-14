import cv2
import numpy as np
import math

class SSLRobot():
    def __init__(
                self,
                id = 0,
                height = 155,
                diameter = 180,
                camera_offset = 90
                ):
        super(SSLRobot, self).__init__()
        self.id = id
        self.height = height
        self.diameter = diameter

        self.position = None
        self.rotation = None
        self.pose_confidence = 0
        self.camera_offset = self.diameter/2

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