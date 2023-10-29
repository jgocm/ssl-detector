import cv2
import math
import numpy as np

class Camera:
    '''
    Defines camera properties
    '''
    def __init__(
                self,
                camera_matrix = np.identity(3),
                camera_distortion=np.zeros((4,1)),
                camera_to_robot_axis_offset = 90,
                camera_height = 175,
                camera_FOV = 78,
                ):

        self.intrinsic_parameters = camera_matrix
        self.distortion_parameters = camera_distortion
        self.rotation_vector: np.array((3,1)).T
        self.rotation_matrix: np.array((3,3))
        self.translation_vector: np.array((3,1)).T
        

        self.height = camera_height                 # IN MILLIMETERS
        self.offset = camera_to_robot_axis_offset   # IN MILLIMETERS
        self.FOV = camera_FOV                       # IN DEGREES

    def compute_pose_from_points(self, points3d, points2d):
        """
        Compute camera pose to object from 2D-3D points correspondences

        Solves PnP problem using OpenCV solvePnP() method assigning
        camera pose from the corresponding 2D-3D matched points.

        Parameters
        ------------
        points3d: 3D coordinates of points

        points2d: pixel positions on image
        """
        
        _,rvec,tvec=cv2.solvePnP(
                                points3d,
                                points2d,
                                self.intrinsic_parameters,
                                self.distortion_parameters
                                )                                

        rmtx, jacobian=cv2.Rodrigues(rvec)
        
        pose = cv2.hconcat((rmtx,tvec))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose)

        camera_position = -np.linalg.inv(rmtx)@tvec
        self.height = camera_position[2,0]

        self.rotation_vector = rvec
        self.rotation_matrix = rmtx
        self.translation_vector = tvec

class ParticleVision:
    '''
    Class for simulating Vision Blackout vision module
    '''

    path_to_intrinsic_parameters = "/home/rc-blackout/rSoccer-MCL/rsoccer_gym/Perception/camera_matrix_C922.txt"
    path_to_points3d = "/home/rc-blackout/rSoccer-MCL/rsoccer_gym/Perception/calibration_points3d.txt"
    path_to_points2d = "/home/rc-blackout/rSoccer-MCL/rsoccer_gym/Perception/calibration_points2d.txt"
    camera_matrix = np.loadtxt(path_to_intrinsic_parameters)
    points3d = np.loadtxt(path_to_points3d, dtype="float64")
    points2d = np.loadtxt(path_to_points2d, dtype="float64")

    def __init__(
                self,
                vertical_lines_nr = 1,
                input_width = 640,
                input_height = 480,
                camera_matrix = camera_matrix,
                points3d = points3d,
                points2d = points2d
                ):
        self.camera = Camera(camera_matrix=camera_matrix)
        self.camera.compute_pose_from_points(points3d=points3d, points2d=points2d)
        self.vertical_lines_nr = vertical_lines_nr
        self.set_detection_angles_random(vertical_lines_nr)    # IN DEGREES  

    def set_detection_angles_uniform(self, vertical_lines_nr):
        vertical_scan_angles = []
        for i in range(0,vertical_lines_nr):
            angle = (i+1)*self.camera.FOV/(vertical_lines_nr+1) - self.camera.FOV/2
            vertical_scan_angles.append(angle)
        self.vertical_scan_angles = vertical_scan_angles

    def set_detection_angles_random(self, vertical_lines_nr):
        vertical_scan_angles = []
        for i in range(0,vertical_lines_nr):
            angle = np.random.uniform(-self.camera.FOV/2, self.camera.FOV/2)
            vertical_scan_angles.append(angle)
        self.vertical_scan_angles = vertical_scan_angles

    def set_detection_angles_from_list(self, vertical_angles_list = [0]):
        self.vertical_scan_angles = vertical_angles_list

    def project_line(self, x, y, theta):
        coef = math.tan(math.radians(theta))
        intercept = y - coef*x
        return coef, intercept

    def intercept_left_boundary(self, a, b, field):
        x = field.x_min
        y = a*x + b
        return x, y

    def intercept_right_boundary(self, a, b, field):
        x = field.x_max
        y = a*x + b
        return x, y

    def intercept_lower_boundary(self, a, b, field):
        y = field.y_min
        x = (y-b)/a
        return x, y

    def intercept_upper_boundary(self, a, b, field):
        y = field.y_max
        x = (y-b)/a
        return x, y

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)     

    def intercept_field_boundaries_optimized(self, x, y, line_dir, field):
        # CHECK IF THETA RESULTS IN 0's
        dangerous_theta = (line_dir % 9 == 0 and line_dir % 10 == 0)
        SMALL_VALUE = 1e-4

        # FIND INTERSECTION VECTORS
        theta = np.deg2rad(line_dir)
        if not dangerous_theta:
            v = np.array([
                (field.x_min-x)/np.cos(theta),
                (field.x_max-x)/np.cos(theta),
                (field.y_min-y)/np.sin(theta),
                (field.y_max-y)/np.sin(theta)
            ])
            min_v = np.min(v[v > 0])
        else:
            v = np.array([
                (field.x_min-x)*np.cos(theta),
                (field.x_max-x)*np.cos(theta),
                (field.y_min-y)*np.sin(theta),
                (field.y_max-y)*np.sin(theta)
            ])
            min_v = np.min(v[v > SMALL_VALUE])
        return x+min_v*np.cos(theta), y+min_v*np.sin(theta)

    def intercept_field_boundaries(self, x, y, line_dir, field):
        a, b = self.project_line(x, y, line_dir)
        if 0 == line_dir:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            return x1, y1
        elif 90 == line_dir:
            x1, y1 = self.intercept_upper_boundary(a, b, field)
            return x1, y1
        elif 180 == line_dir or -180 == line_dir:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            return x1, y1
        elif -90 == line_dir:
            x1, y1 = self.intercept_lower_boundary(a, b, field)
            return x1, y1
        elif 0 < line_dir and line_dir < 90:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            x2, y2 = self.intercept_upper_boundary(a, b, field)
        elif 90 < line_dir and line_dir < 180:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            x2, y2 = self.intercept_upper_boundary(a, b, field)
        elif -180 < line_dir and line_dir < -90:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            x2, y2 = self.intercept_lower_boundary(a, b, field)
        elif -90 < line_dir and line_dir < 0:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            x2, y2 = self.intercept_lower_boundary(a, b, field)
        else:
            return -1, -1
        
        if self.get_distance(x, y, x1, y1) < self.get_distance(x, y, x2, y2):
            return x1, y1
        else:
            return x2, y2

    def convert_xy_to_angles(self, x, y):
        '''
        Converts an x, y relative position to relative vertical and horizontal angles, as suggested in: 
            Fast and Robust Edge-Based Localization in the Sony Four-Legged Robot League - 2003
        '''
        theta_v = np.rad2deg(np.arctan2(self.camera.height, x))
        theta_h = np.rad2deg(np.arctan2(y, x))
        return [theta_v, theta_h]

    def convert_xy_to_polar(self, x, y):
        '''
        Converts an x, y relative position to relative polar coordinates (distance and bearing angle), as suggested in: 
            Monte Carlo Localization for Robocup 3D Soccer Simulation League - 2016
        '''
        distance = np.sqrt(x**2 + y**2)
        theta = np.rad2deg(np.arctan2(y, x))
        return [distance, theta]

    def detect_boundary_points(self, x, y, w, field):
        intercepts = []
        SMALL_VALUE = 1e-7
        for angle in self.vertical_scan_angles:
            line_dir = angle + w
            line_dir = ((line_dir + 180) % 360) - 180
            interception_x, interception_y = self.intercept_field_boundaries(x, y, line_dir, field)
            interception_x, interception_y = self.intercept_field_boundaries_optimized(x, y, line_dir, field)
            interception_x, interception_y = self.convert_to_local(interception_x, interception_y, x, y, w)
            intercepts.append(self.convert_xy_to_polar(interception_x, interception_y))

        return intercepts

    def detect_boundary_points_random(self, x, y, w, field):
        intercepts = []
        self.set_detection_angles_random(self.vertical_lines_nr)
        for angle in self.vertical_scan_angles:
            line_dir = angle + w
            line_dir = ((line_dir + 180) % 360) - 180
            interception_x, interception_y = self.intercept_field_boundaries(x, y, line_dir, field)
            interception_x, interception_y = self.convert_to_local(interception_x, interception_y, x, y, w)
            intercepts.append(self.convert_xy_to_polar(interception_x, interception_y))

        return intercepts

    def get_robot_to_positive_goal_vector(self, x, y, field):
        goal_x, goal_y = field.x_max - field.boundary_width, 0
        return goal_x - x, goal_y - y

    def get_robot_to_negative_goal_vector(self, x, y, field):
        goal_x, goal_y = field.x_min + field.boundary_width, 0
        return goal_x - x, goal_y - y


    def limit_angle_degrees(self, angle):
        while angle>180:
            angle -= 2*180
        while angle<-180:
            angle += 2*180
        return angle

    def track_positive_goal_center(self, x, y, w, field):
        x, y = self.get_robot_to_positive_goal_vector(x, y, field)
        distance = np.sqrt(x**2 + y**2)
        local_angle = self.limit_angle_degrees(np.rad2deg(np.arctan2(y, x)) - w)
        if np.abs(local_angle)>30: has_goal = 0
        else: has_goal = 1
        return has_goal, distance, local_angle

    def track_negative_goal_center(self, x, y, w, field):
        x, y = self.get_robot_to_negative_goal_vector(x, y, field)
        distance = np.sqrt(x**2 + y**2)
        local_angle = self.limit_angle_degrees(np.rad2deg(np.arctan2(y, x)) - w)
        if np.abs(local_angle)>30: has_goal = 0
        else: has_goal = 1

        return has_goal, distance, local_angle

    def convert_to_local(self, global_x, global_y, robot_x, robot_y, robot_w):
        x = global_x - robot_x
        y = global_y - robot_y
        robot_w = math.radians(robot_w)
        x, y = x*np.cos(robot_w) + y*np.sin(robot_w),\
            -x*np.sin(robot_w) + y*np.cos(robot_w)

        return x, y

#TODO: REMOVE FIELD CLASS FROM THIS FILE
class Field():
    def __init__(
                self,
                field_width = 3.760,
                field_length = 5.640,
                penalty_area_width = 2.000,
                penalty_area_depth = 1.000,
                center_radius = 1.000,
                boundary_width = 0.180,
                line_thickness = 0.020
                ):
                
        # FIELD DIMENSION PROPERTIES
        self.width = field_width
        self.length = field_length
        self.penalty_area_width = penalty_area_width
        self.penalty_area_depth = penalty_area_depth
        self.boundary_width = boundary_width
        self.line_thickness = line_thickness
        self.center_radius = center_radius

        # FIELD LIMITS
        self.x_max = field_length/2 + boundary_width
        self.x_min = -self.x_max
        self.y_max = field_width/2 + boundary_width
        self.y_min = -self.y_max

    def redefineFieldLimits(self, x_max, y_max, x_min, y_min):
        self.x_max = x_max
        self.y_max = y_max
        self.x_min = x_min
        self.y_min = y_min

if __name__ == "__main__":
    field = Field()
        
    field.boundary_width = 0.3
    field.x_max = 4.2
    field.x_min = -0.3
    field.y_max = 3
    field.y_min = -3

    robot_x, robot_y, robot_w = 0, 0, 15

    vision = ParticleVision(vertical_lines_nr=1)
    
    has_goal, distance, local_angle = vision.track_positive_goal_center(robot_x, robot_y, robot_w, field)
    print(has_goal, distance, local_angle)
    # import pdb;pdb.set_trace()
