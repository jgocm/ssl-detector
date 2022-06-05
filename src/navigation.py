from turtle import distance
import CommTypes_pb2 as pb
import math
import numpy as np

class GroundPoint():
    def __init__(
                self,
                x = 0,
                y = 0
                ):
        self.x = x
        self.y = y
    
    def setPosition(self, x, y):
        self.x = x
        self.y = y

    def getDistance(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def getDirection(self):
        w = math.atan2(self.y, self.x)
        return w
    
    def getVector(self):
        return np.array([self.x, self.y])

class TargetPoint(GroundPoint):
    def __init__(
                self,
                x = 0,
                y = 0,
                w = 0
                ):
        super().__init__(self, x, y)
        self.w = w
        self.type = pb.protoPositionSSL.unknown

    def get2XYCoordinatesVector(self, x1, y1, x2, y2):
        """
        Returns distance from (x1, y1) to (x2, y2) and the vector direction regarding the origin
        x1: x coordinate of point 1
        y1: y coordinate of point 1
        x2: x coordinate of point 2
        y2: y coordinate of point 2
        -------------------
        returns:
        distance: norm of vector p1->p2 (p2-p1)
        direction: 2D numpy array directional vector of p1->p2 (p2-p1)
        angle: rotation angle of directional vector in radians
        """
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        distance, direction, angle = self.get2PointsVector(p1, p2)

        return distance, direction, angle

    def get2PointsVector(self, p1, p2):
        """
        Returns distance from p1 to p2 and the vector direction regarding the origin
        p1: 2D numpy array ([x, y])
        p2: 2D numpy array ([x, y])
        -------------------
        returns:
        distance: norm of vector p1->p2 (p2-p1)
        direction: 2D numpy array directional vector of p1->p2 (p2-p1)
        angle: rotation angle of directional vector in radians
        """
        v = p2 - p1
        distance = np.linalg.norm(v)
        direction = v/distance
        angle = math.atan(v[1]/v[0])

        return distance, direction, angle
    
    def getRotatioMatrix(self, theta):
        """
        Computes rotation matrix from given angle in the form:
        | cos(theta)    -sin(theta)|
        | sin(theta)    cos(theta) |

        theta: rotation angle in radians
        -------------------
        returns:
        R_theta: rotation matrix for theta
        """
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        R_theta = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        return R_theta
    
    def rotateVector(self, v, relative_angle):
        """
        Rotates a vector in counter clockwise direction around a given angle
        
        v: vector to rotate
        theta: rotation angle in radians
        -------------------
        returns:
        v_rotated: vector after rotation
        """
        R = self.getRotatioMatrix(relative_angle)
        v_rotated = R@v

        return v_rotated
    
    def get2DPointRelativeToLinePoint(self, p1 = np.array([0,0]), p2 = np.array([0,0]), relative_angle = 0, relative_distance = 0):
        """
        Computes x,y coordinates relative to the line p1->p2 with a given distance to p2:
        XY = R(relative_angle)@(p2-p1)*realtive_distance/norm(p2-p1) + p2

        p1: initial point of the line
        p2: final point of line and reference point
        relative_angle: direction of target point relative to line p1->p2
        relative_distance: distance of target point from p2
        -------------------
        returns:
        XY[0]: x coordinate
        XY[1]: y coordinate
        """
        _, direction_vector, _ = self.get2PointsVector(p1, p2)
        v_rotated = self.rotateVector(direction_vector, relative_angle)
        XY = p2 + v_rotated*relative_distance

        return XY[0], XY[1]
    
    def get2DPointRelativeToLine2DCoordinates(self, x1 = 0, y1 = 0, x2 = 0, y2 = 0, relative_angle = 0, relative_distance = 0):
        """
        Computes x,y coordinates relative to the line p1->p2 with a given distance to p2:
        XY = R(relative_angle)@(p2-p1)*realtive_distance/norm(p2-p1) + p2

        x1: x coordinate from initial point of the line
        y1: y coordinate from initial point of the line
        x2: x coordinate from final point of line and reference point
        y2: y coordinate from final point of line and reference point
        relative_angle: direction of target point relative to line p1->p2
        relative_distance: distance of target point from p2
        -------------------
        returns:
        XY[0]: x coordinate
        XY[1]: y coordinate
        """
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        x, y = self.getTargetRelativeToLinePoint(p1, p2, relative_angle, relative_distance)
        
        return x, y
    
    def getTargetCoordinatesRelativeToLinePoint(self, p1 = np.array([0,0]), p2 = np.array([0,0]), relative_angle = 0, relative_distance = 0):
        """
        Computes x,y coordinates and direction relative to the line p1->p2 with a given distance to p2:
        XY = R(relative_angle)@(p2-p1)*realtive_distance/norm(p2-p1) + p2

        p1: initial point of the line
        p2: final point of line and reference point
        relative_angle: direction of target point relative to line p1->p2
        relative_distance: distance of target point from p2
        -------------------
        returns:
        target_x: target x coordinate
        target_y: target y coordinate
        target_w: target direction pointing to p2
        """
        _, direction_vector, _ = self.get2PointsVector(p1, p2)
        v_rotated = self.rotateVector(direction_vector, relative_angle)
        XY = p2 + v_rotated*relative_distance
        target_x, target_y, target_w = XY[0], XY[1], math.atan(direction_vector[1]/direction_vector[0])

        return target_x, target_y, target_w

    def getTargetRelativeToLine2DCoordinates(self, x1 = 0, y1 = 0, x2 = 0, y2 = 0, relative_angle = 0, relative_distance = 0):
        """
        Computes x,y coordinates and direction relative to the line p1->p2 with a given distance to p2:
        XY = R(relative_angle)@(p2-p1)*realtive_distance/norm(p2-p1) + p2

        x1: x coordinate from initial point of the line
        y1: y coordinate from initial point of the line
        x2: x coordinate from final point of line and reference point
        y2: y coordinate from final point of line and reference point
        relative_angle: direction of target point relative to line p1->p2
        relative_distance: distance of target point from p2
        -------------------
        returns:
        target: (target x coordinate, target y coordinate, target direction pointing to p2)
        """
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        target_x, target_y, target_w = self.getTargetCoordinatesRelativeToLinePoint(p1, p2, relative_angle, relative_distance)
        target = TargetPoint(target_x, target_y, target_w)
        
        return target