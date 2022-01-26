import cv2
import numpy as np
import Jetson.GPIO as GPIO

class robot():
    def __init__(self):
        super(robot, self).__init__()
        self.id = None
        self.position = None
        self.rotation = None
        self.pose_confidence = 0
        self.pinMode = GPIO.BOARD
    
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

