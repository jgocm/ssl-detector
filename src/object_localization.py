from re import A
import numpy as np
import cv2
import math
import time
from sklearn import linear_model

class KeypointRegression():
    def __init__(
            self
            ):
        # DEFINE COLORS:
        self.BLACK = [0, 0, 0]
        self.BLUE = [255, 0, 0]
        self.GREEN = [0, 255, 0]
        self.RED = [0, 0, 255]
        self.WHITE = [255, 255, 255]
        self.min_line_length = 3
        self.skip_frame = True

    def ballAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.2):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def ballAsPointLinearRegression(self, left, top, right, bottom, weight_x, weight_y):
        x = [left, right, top, bottom, 1]@weight_x
        y = [left, right, top, bottom, 1]@weight_y
        return x, y
    
    def robotAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.1):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def goalCenterAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.1):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def transformPointsToRegressionData(self, line_regression_points):
        df = np.array(line_regression_points)
        X = df[:,0]
        X = X[:, np.newaxis]
        y = df[:,1]
        y = y[:, np.newaxis]

        return X, y

    def makeLinearRegressionModel(self, line_regression_points):
        try:
            X, y = self.transformPointsToRegressionData(line_regression_points)
            model = linear_model.LinearRegression().fit(X=X, y=y)
            return model.coef_, model.intercept_
        except:
            self.skip_frame = True
            return -1, -1

    def makeRANSACRegressionModel(self, line_regression_points):
        try:
            X, y = self.transformPointsToRegressionData(line_regression_points)
            model = linear_model.RANSACRegressor().fit(X=X, y=y)
            return model.estimator_.coef_, model.estimator_.intercept_
        except:
            self.skip_frame = True
            return -1, -1

    def goalLineDetection(self, src, left, top, right, bottom):
        """
        Make descripition here
        """
        # make copy from source image for segmentation
        segmented_img = src.copy()

        # compute bounding box width and height
        height, width = bottom-top, right-left

        # line scans offset
        vertical_lines_offset = int(0.05 * width)

        # points used for linear regression
        goal_line_points = []

        for line_x in range(left, right, vertical_lines_offset):
            # segment vertical lines
            for pixel_y in range(top, bottom):
                blue, green, red = src[pixel_y, line_x]
                if green > 120 and red < 110:
                    color = self.GREEN
                elif green < 50 and red < 50 and blue < 50:
                    color = self.BLACK
                else:
                    color = self.WHITE
                segmented_img[pixel_y, line_x] = color
            
            # detect line points from edges
            goal_line = False
            kernel = [-1, 1]
            line_points = []
            for pixel_y in range(bottom, int((top+bottom)/2), -1):
                blue = segmented_img[pixel_y-1:pixel_y+1, line_x][:,0]
                blue_gradient = blue@kernel
                if blue_gradient < -200 and goal_line == False:
                    goal_line = True
                elif blue_gradient > 200 and goal_line == True:
                    goal_line = False
                    # if more than 3 consecutive points are detected, it is probably not the goal line
                    if len(line_points)<3:
                        for point in line_points:
                            goal_line_points.append(point)
                    break
                if goal_line == True:
                    line_points.append([line_x, pixel_y])

        return goal_line_points
    
    def goalLineRegression(self, src, left, top, right, bottom):
        try:
            goal_line_points = self.goalLineDetection(src, left, top, right, bottom)
        except:
            self.skip_frame = True
            goal_line_points = []
        
        goal_line_model = self.makeLinearRegressionModel(goal_line_points)

        return goal_line_model
     

    def goalLeftPostDetectionGrayScale(self, src, left, top, right, bottom):
        """
        Make descripition here
        """
        # compute bounding box width and height
        height, width = bottom-top, right-left

        # line scans offset
        horizontal_lines_offset = int(0.05 * height)

        # convert img to gray scale
        posts_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

        # points used for line regression
        goal_post_points = []

        for line_y in range(top, bottom, int(horizontal_lines_offset)):
            kernel = [-0.75, -1, 0, 1, 0.75]
            left_x = left
            for pixel_x in range(left, right):
                pixel_color = posts_img[line_y, pixel_x]
                if pixel_color>180:
                    gradient = posts_img[line_y, pixel_x-2:pixel_x+3]@kernel
                    if np.abs(gradient)>20 and np.abs(gradient)<70:
                        if pixel_x>left_x:
                            left_x = pixel_x
            if left_x != left:
                # x coordinate will be regressed from y coordinate
                # so data should be in the form (y, x) for linear regression
                goal_post_points.append([line_y, left_x])
        
        return goal_post_points
    
    def goalRightPostDetectionGrayScale(self, src, left, top, right, bottom):
        """
        Make descripition here
        """
        # compute bounding box width and height
        height, width = bottom-top, right-left

        # line scans offset
        horizontal_lines_offset = int(0.05 * height)

        # convert img to gray scale
        posts_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

        # points used for line regression
        goal_post_points = []

        for line_y in range(top, bottom, int(horizontal_lines_offset)):
            kernel = [-0.75, -1, 0, 1, 0.75]
            right_x = right
            for pixel_x in range(left, right):
                pixel_color = posts_img[line_y, pixel_x]
                if pixel_color>180:
                    gradient = posts_img[line_y, pixel_x-2:pixel_x+3]@kernel
                    if np.abs(gradient)>20 and np.abs(gradient)<70:
                        if pixel_x<right_x:
                            right_x = pixel_x
            if right_x != right:
                # x coordinate will be regressed from y coordinate
                # so data should be in the form (y, x) for linear regression
                goal_post_points.append([line_y, right_x])
        
        return goal_post_points

    def goalLeftPostRegressionFromLeft(self, src, left, top, right, bottom, left_to_right_proportion = 0.5):
        right = int(left_to_right_proportion*(left+right))
        try:
            goal_post_points = self.goalLeftPostDetectionGrayScale(src, left, top, right, bottom)
        except:
            goal_post_points = []
            self.skip_frame = True

        goal_post_model = self.makeRANSACRegressionModel(goal_post_points)

        return goal_post_model
    
    def goalRightPostRegressionFromRight(self, src, left, top, right, bottom, left_to_right_proportion = 0.5):
        left = int((1-left_to_right_proportion)*(left+right))
        try:
            goal_post_points = self.goalRightPostDetectionGrayScale(src, left, top, right, bottom)
        except:
            goal_post_points = []
            self.skip_frame = True

        goal_post_model = self.makeRANSACRegressionModel(goal_post_points)
        return goal_post_model

    def goalLeftPostDetectionBGR(self, src, left, top, right, bottom):
        """
        Make descripition here
        """
        # make copy from source image for segmentation
        segmented_img = src.copy()

        # compute bounding box width and height
        height, width = bottom-top, right-left

        # line scans offset
        horizontal_lines_offset = int(0.05 * height)

        # points used for line regression
        goal_post_points = []

        for line_y in range(top, bottom, horizontal_lines_offset):
            # segment image horizontal line
            for pixel_x in range(left, right):
                blue, green, red = src[line_y, pixel_x]
                # paint strong white pixels with white -> probably not a post
                if blue > 170 and green > 170 and red > 170:
                    color = self.WHITE
                # paint darker pixels with black
                elif blue < 50 and green < 50 and red < 50:
                    color = self.BLACK
                else:
                    color = self.RED
                segmented_img[line_y, pixel_x] = color

            # find post from edge detection
            goal_post = False
            kernel = [1, -1]
            post_points = []
            for pixel_x in range(right, left, -1):
                red = segmented_img[line_y, pixel_x-1:pixel_x+1][:,2]
                red_gradient = red@kernel
                blue = segmented_img[line_y, pixel_x-1:pixel_x+1][:,0]
                blue_gradient = blue@kernel
                if red[-1] == 255 and blue_gradient < -200 and goal_post == False:
                    goal_post = True
                elif red_gradient < -200 and goal_post == True:
                    goal_post = False
                    if len(post_points)<5:
                        for pixel in post_points:
                            goal_post_points.append(pixel)
                    break
                if goal_post == True:
                    # x coordinate will be regressed from y coordinate
                    # so data should be in the form (y, x) for linear regression
                    post_points.append([line_y, pixel_x])
            
        return goal_post_points

    def goalRightPostDetectionBGR(self, src, left, top, right, bottom):
        """
        Make descripition here
        """
        # make copy from source image for segmentation
        segmented_img = src.copy()

        # compute bounding box width and height
        height, width = bottom-top, right-left

        # line scans offset
        horizontal_lines_offset = int(0.05 * height)

        # points used for line regression
        goal_post_points = []

        for line_y in range(top, bottom, horizontal_lines_offset):
            # segment image horizontal line
            for pixel_x in range(left, right):
                blue, green, red = src[line_y, pixel_x]
                # paint strong white pixels with white -> probably not a post
                if blue > 170 and green > 170 and red > 170:
                    color = self.WHITE
                # paint darker pixels with black
                elif blue < 50 and green < 50 and red < 50:
                    color = self.BLACK
                else:
                    color = self.RED
                segmented_img[line_y, pixel_x] = color

            # find post from edge detection
            goal_post = False
            kernel = [-1, 1]
            post_points = []
            for pixel_x in range(left, right, 1):
                red = segmented_img[line_y, pixel_x-1:pixel_x+1][:,2]
                red_gradient = red@kernel
                blue = segmented_img[line_y, pixel_x-1:pixel_x+1][:,0]
                blue_gradient = blue@kernel
                if red[-1] == 255 and blue_gradient < -200 and goal_post == False:
                    goal_post = True
                elif red_gradient < -200 and goal_post == True:
                    goal_post = False
                    if len(post_points)<5:
                        for pixel in post_points:
                            goal_post_points.append(pixel)
                    break
                if goal_post == True:
                    # x coordinate will be regressed from y coordinate
                    # so data should be in the form (y, x) for linear regression
                    post_points.append([line_y, pixel_x])
            
        return goal_post_points

    def goalLeftPostRegressionFromRight(self, src, left, top, right, bottom, left_to_right_proportion = 0.5):
        right = int(left_to_right_proportion*(left+right))
        try:
            goal_post_points = self.goalLeftPostDetectionBGR(src, left, top, right, bottom)
        except:
            goal_post_points = []
            self.skip_frame = True        
        goal_post_model = self.makeLinearRegressionModel(goal_post_points)
        return goal_post_model
    
    def goalRightPostRegressionFromLeft(self, src, left, top, right, bottom, left_to_right_proportion = 0.5):
        left = int((1-left_to_right_proportion)*(left+right))
        try:
            goal_post_points = self.goalRightPostDetectionBGR(src, left, top, right, bottom)
        except:
            goal_post_points = []
            self.skip_frame = True        
        goal_post_model = self.makeLinearRegressionModel(goal_post_points)
        return goal_post_model

    def goalLeftPostRegressionFromMiddle(self, src, left, top, right, bottom, left_to_right_proportion = 0.5):
        return self.goalLeftPostRegressionFromRight(src, left, top, right, bottom, left_to_right_proportion)

    def goalRightPostRegressionFromMiddle(self, src, left, top, right, bottom, left_to_right_proportion = 0.5):
        return self.goalRightPostRegressionFromLeft(src, left, top, right, bottom, left_to_right_proportion)
    
    def getGoalSide(self, angular_coef):
        # decides wether the robot is on the right or the left of the goal from goal line angular coefficient
        if np.abs(angular_coef) < 0.01:
            side = "middle"
        elif angular_coef < 0:
            side = "left"
        else:
            side = "right"
        
        return side

    def goalPostsRegresion(self, src, left, top, right, bottom, left_to_right_proportion = 0.5, side = "middle"):
        if side == "middle":
            goal_left_post = self.goalLeftPostRegressionFromMiddle(src, left, top, right, bottom, left_to_right_proportion)
            goal_right_post = self.goalRightPostRegressionFromMiddle(src, left, top, right, bottom, left_to_right_proportion)

        elif side == "left":
            goal_left_post = self.goalLeftPostRegressionFromLeft(src, left, top, right, bottom, left_to_right_proportion)
            goal_right_post = self.goalRightPostRegressionFromRight(src, left, top, right, bottom, left_to_right_proportion)

        elif side == "right":
            goal_left_post = self.goalLeftPostRegressionFromRight(src, left, top, right, bottom, left_to_right_proportion)
            goal_right_post = self.goalRightPostRegressionFromLeft(src, left, top, right, bottom, left_to_right_proportion)
        
        return goal_left_post, goal_right_post

    def linesIntersection(self, a1, b1, a2, b2):
        x = (b2-b1)/(a1-a2)
        y = a1*x+b1
        return x, y

    def goalCornersRegression(self, src, left, top, right, bottom, left_to_right_proportion = 0.5):
        goal_line_coef, goal_line_intercept = self.goalLineRegression(src, left, top, right, bottom)

        side = self.getGoalSide(goal_line_coef)
        goal_left_post, goal_right_post = self.goalPostsRegresion(src, left, top, right, bottom, left_to_right_proportion, side)
    
        height, width = int(src.shape[0]), int(src.shape[1])
        if self.skip_frame == False:
            left_corner = self.linesIntersection(
                                    goal_line_coef, 
                                    goal_line_intercept,
                                    1/(goal_left_post[0]+0.001),
                                    -goal_left_post[1]/(goal_left_post[0]+0.001))

            right_corner = self.linesIntersection(
                                    goal_line_coef, 
                                    goal_line_intercept,
                                    1/(goal_right_post[0]+0.001),
                                    -goal_right_post[1]/(goal_right_post[0]+0.001))
            
            return left_corner, right_corner
        else:
            return -1, -1

    def goalAsCorners(self, src, left, top, right, bottom):
        self.skip_frame = False
        left_to_right_proportion = 0.5
        left_corner, right_corner = self.goalCornersRegression(src, left, top, right, bottom, left_to_right_proportion)
        return left_corner, right_corner

    def goalAsLine(self, src, left, top, right, bottom):
        self.skip_frame = False
        coef, intercept = self.goalLineRegression(src, left, top, right, bottom)
        return coef, intercept

class Camera():
    def __init__(
                self,
                camera_matrix=np.identity(3),
                camera_distortion=np.zeros((4,1)),
                camera_initial_position=np.zeros(3),
                vision_offset=np.zeros(3)
                ):
        super(Camera, self).__init__()
        self.intrinsic_parameters = camera_matrix
        self.distortion_parameters = camera_distortion

        self.rotation_vector = np.zeros((3,1))
        self.rotation_matrix = np.zeros((3,3))
        self.translation_vector = np.zeros((3,1)).T

        self.position = np.zeros((3,1))
        self.rotation = np.zeros((3,1)) # EULER ROTATION ANGLES
        self.height = 0

        self.initial_position = camera_initial_position
        # apply XYW offset if calibration ground truth position is known
        self.offset = vision_offset

    def setIntrinsicParameters(self, mtx):
        if np.shape(mtx)==(3,3): 
            self.intrinsic_parameters = mtx
            print(f"Camera Matrix is:")
            print(mtx)
            return self
        else:
            print(f"Camera Matrix must be of shape (3,3) and the inserted matrix has shape {np.shape(mtx)}")
            return self
    
    def setDistortionParameters(self, dist):
        if np.shape(dist)==(4,1):
            self.distortion_parameters = dist
            print("Camera distortion parameters are:")
            print(dist)
            return True
        else:
            print(f"Camera distortion parameters must be of shape (4,1) and the inserted array has shape {np.shape(dist)}")
            return False
    
    def setOffset(self, offset):
        if np.shape(offset)==(3,):
            self.offset = offset
            print(f"Position offset is:")
            print(offset)
        else:
            print(f"Position offset must be of shape (3,) and the inserted matrix has shape {np.shape(offset)}")
    
    def fixPoints3d(self, points3d):
        return points3d-self.initial_position

    def computePoseFromPoints(self, points3d, points2d):
        """
        Compute camera pose to object from 2D-3D points correspondences

        Solves PnP problem using OpenCV solvePnP() method assigning
        camera pose from the corresponding 2D-3D matched points.

        Parameters
        ------------
        points3d: 3D coordinates of points

        points2d: pixel positions on image
        """
        #points3d = self.fixPoints3d(points3d=points3d)
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
        height = camera_position[2,0]
        self.offset = (camera_position.T-self.initial_position).T

        self.rotation_vector = rvec
        self.rotation_matrix = rmtx
        self.translation_vector = tvec

        self.position = camera_position
        self.rotation = euler_angles
        self.height = height

        return camera_position, euler_angles
    
    def computeRotationMatrixFromAngles(self, euler_angles):
        theta_x = euler_angles[0][0]
        theta_x = math.radians(theta_x)
        cx = math.cos(theta_x)
        sx = math.sin(theta_x)
        rX_cam = np.array([
            [1,0,0],
            [0,cx,-sx],
            [0,sx,cx]
        ])

        theta_y = euler_angles[1][0]
        theta_y = math.radians(theta_y)
        cy = math.cos(theta_y)
        sy = math.sin(theta_y)
        rY_cam = np.array([
            [cy,0,sy],
            [0,1,0],
            [-sy,0,cy]
        ])

        theta_z = euler_angles[2][0]
        theta_z = math.radians(theta_z)
        cz = math.cos(theta_z)
        sz = math.sin(theta_z)
        rZ_cam = np.array([
            [cz,-sz,0],
            [sz,cz,0],
            [0,0,1]
        ])

        rmtx = rZ_cam @ rY_cam @ rX_cam
        return rmtx

    def setPoseFromFile(self, camera_position, euler_angles):
        self.position = camera_position
        self.rotation = euler_angles
        self.height = camera_position[2,0]

        rmtx = self.computeRotationMatrixFromAngles(euler_angles)
        self.rotation_matrix = rmtx

        tvec = -np.matmul(np.linalg.inv(rmtx),np.matrix(camera_position))

        return tvec, rmtx

    def pixelToCameraCoordinates(self, x, y, z_world=0):
        uvPoint = np.array([(x,y,1)])
        mtx = self.intrinsic_parameters
        rmtx = self.rotation_matrix
        height = self.height
        
        leftsideMat = np.linalg.inv(rmtx)@(np.linalg.inv(mtx)@np.transpose(uvPoint))
        s = -(height-z_world)/leftsideMat[2]
        
        p = s*leftsideMat

        return p

    def ballAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.15):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def ballAsPointLinearRegression(self, left, top, right, bottom, weight_x, weight_y):
        x = [left, right, top, bottom, 1]@weight_x
        y = [left, right, top, bottom, 1]@weight_y
        return x, y
    
    def robotAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.1):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def goalAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.1):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y

    def cameraToPixelCoordinates(self, x, y, z_world=0):
        M = self.intrinsic_parameters
        R = self.rotation_matrix
        t = self.translation_vector
        height = self.height
        cameraPoint = np.array([(x,y,z_world-height)])

        rightSideMat = M@(R@(cameraPoint).T)

        s = rightSideMat[2]

        uvPoint = rightSideMat/s

        return uvPoint
    
    def selfLocalizationFromGoalCorners(self, x1, y1, x2, y2):
        theta = math.atan((y2-y1)/(x1-x2))
        
        x0 = (x1 + x2)/2
        y0 = (y1 + y2)/2

        center_x = 2820
        center_y = 0

        xt = center_x - math.cos(theta)*x0 + math.sin(theta)*y0
        yt = center_y - math.sin(theta)*x0 - math.cos(theta)*y0

        return xt, yt, theta

    def selfOrientationFromGoalLine(self, a, b):
        R = self.rotation_matrix
        K = self.intrinsic_parameters
        M = np.linalg.inv(R)@np.linalg.inv(K)
        r = np.array([0, 0, 1])@M
        d = np.array([1, a, 0])
        p0 = np.array([0, b, 1])
        diff = (M@np.transpose((r@np.transpose(d))*p0-(r@np.transpose(p0))*d))

        theta = -math.atan(diff[1]/diff[0])
        
        return theta

if __name__=="__main__":
    import object_detection
    import tensorrt as trt
    import interface
    import os
    
    cwd = os.getcwd()

    # START TIME
    start = time.time()

    # SET WINDOW TITLE
    WINDOW_NAME = 'Object Localization'

    # USB CAMERA SETUP
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # OBJECT DETECTION MODEL
    trt_net = object_detection.DetectNet(
                model_path=cwd+"/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt", 
                labels_path=cwd+"/models/ssl_labels.txt", 
                input_width=300, 
                input_height=300,
                score_threshold = 0.25,
                draw = True,
                display_fps = False,
                TRT_LOGGER = trt.Logger(trt.Logger.INFO)
                )
    trt_net.loadModel()
    regression_weights = np.loadtxt(cwd+f"/models/regression_weights.txt")

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = Camera(
                camera_matrix=camera_matrix,
                camera_initial_position=calibration_position
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # USER INTERFACE SETUP
    myGUI = interface.GUI(
                        play = True,
                        mode = "detection"
                        )
    nr = 382
    frame_has_goal = False

    def line(x1, y1, x2, y2):
        a = (y2-y1)/(x2-x1)
        b = y2-a*x2
        return a, b

    while True:
        img = cv2.imread(f"/home/vision-blackout/ssl-detector/data/stage2_1/frame{nr}.jpg")
        if cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               print("Check video capture path")
               break
           else:
            myGUI.updateGUI(img)
        
        detections = trt_net.inference(myGUI.screen).detections

        for detection in detections:
            class_id, score, xmin, xmax, ymin, ymax = detection
            
            # BALL LOCALIZATION ON IMAGE
            if class_id==1:     # ball
                # COMPUTE PIXEL FOR BALL POSITION                                                            
                ball_pixel_x, ball_pixel_y = ssl_cam.ballAsPoint(
                                                    left = xmin, 
                                                    top = ymin, 
                                                    right = xmax, 
                                                    bottom = ymax,
                                                    weight_x = 0.5,
                                                    weight_y = 0.15)
                cv2.arrowedLine(myGUI.screen,
                            (int(240), int(640)),
                            (int(ball_pixel_x), int(ball_pixel_y)),
                            (0,0,0),
                            2)
                # DRAW OBJECT POINT ON SCREEN
                myGUI.drawCrossMarker(myGUI.screen, 
                                int(ball_pixel_x), 
                                int(ball_pixel_y),
                                text_size = 0.6)
        
            # GOAL LOCALIZATION ON IMAGE
            if class_id==2:     # goal
                # COMPUTE PIXEL FOR GOAL POSITION
                frame_has_goal = True                     
                goal_pixel_x, goal_pixel_y = ssl_cam.goalAsPoint(
                                                    left = xmin, 
                                                    top = ymin, 
                                                    right = xmax, 
                                                    bottom = ymax)
                if frame_has_goal:
                    cv2.arrowedLine(myGUI.screen,
                                (int(goal_pixel_x), int(goal_pixel_y)),
                                (int(ball_pixel_x), int(ball_pixel_y)),
                                (0,0,0),
                                2)
                    a, b = line(int(goal_pixel_x), 
                                int(goal_pixel_y),
                                int(ball_pixel_x), 
                                int(ball_pixel_y))
                    cv2.line(myGUI.screen,
                                (int(ball_pixel_x), int(ball_pixel_y)),
                                (int((640-b)/a), int(640)),
                                (0,0,0),
                                2)                    
                    
                # DRAW OBJECT POINT ON SCREEN
                myGUI.drawCrossMarker(myGUI.screen, int(goal_pixel_x), int(goal_pixel_y))

                # BACK PROJECT GOAL POSITION TO CAMERA 3D COORDINATES
                object_position = ssl_cam.pixelToCameraCoordinates(x=goal_pixel_x, y=goal_pixel_y, z_world=0)
                x, y, z = (position[0] for position in object_position)
                caption = f"Position:{x:.2f},{y:.2f}"
                myGUI.drawText(myGUI.screen, caption, (int(goal_pixel_x-25), int(goal_pixel_y+30)), 0.6)
            
        # BACK PROJECT BALL POSITION TO CAMERA 3D COORDINATES
        object_position = ssl_cam.pixelToCameraCoordinates(x=ball_pixel_x, y=ball_pixel_y, z_world=0)
        x, y, z = (position[0] for position in object_position)
        caption = f"Position:{x:.2f},{y:.2f}"
        myGUI.drawText(myGUI.screen, caption, (int(ball_pixel_x-25), int(ball_pixel_y+30)), 0.6)
        
        # DISPLAY WINDOW
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        run_time = time.time()-start
        #myGUI.drawText(myGUI.screen, f"running time: {run_time:.2f}s", (8, 13), 0.5)
        cv2.imshow(WINDOW_NAME, myGUI.screen)
        
        if key == ord('s'):
            cv2.imwrite(cwd+"/paper.jpg",myGUI.screen)
        elif key == ord('q'):
            print(f"frame nr: {nr}")
            break
        
    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()
