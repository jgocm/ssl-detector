import numpy as np
import cv2
import math

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
    
    def cameraToRobotCoordinates(self, x, y, camera_offset=90):
        """
        Converts x, y ground position from camera axis to robot axis
        
        Parameters:
        x: x position from camera coordinates in millimeters
        y: y position from camera coordinates in millimeters
        camera_offset: camera to robot center distance in millimeters
        -----------
        Returns:
        robot_x: x position from robot coordinates in meters
        robot_y: y position from robot coordinates in meters
        robot_w: direction from x, y coordinates in radians
        """
        robot_x = (y + camera_offset)/1000
        robot_y = -x/1000
        robot_w = math.atan2(robot_y, robot_x)

        return robot_x, robot_y, robot_w

    def pixelToRobotCoordinates(self, pixel_x, pixel_y, z_world):
        # BACK PROJECT OBJECT POSITION TO CAMERA 3D COORDINATES
        object_position = self.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
        x, y = object_position[0], object_position[1]

        # CONVERT COORDINATES FROM CAMERA TO ROBOT AXIS
        x, y, w = self.cameraToRobotCoordinates(x[0], y[0])
        return x, y, w

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
    import os
    import interface

    cwd = os.getcwd()

    # SET WINDOW TITLE
    WINDOW_TITLE = 'Object Localization'

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    cam = Camera(camera_matrix=camera_matrix)

    # SET CAMERA EXTRINSIC PARAMETERS FROM 2D<=>3D POINTS CORRESPONDENCE
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # READ IMAGE
    PATH_TO_IMG = cwd+"/configs/calibration_image.jpg"
    img = cv2.imread(PATH_TO_IMG)

    # USER INTERFACE SETUP
    myGUI = interface.GUI(play = True,
                          mode = "debug")

    while True:
        # Run UI
        myGUI.runUI(myGUI.screen)

        # DISPLAY WINDOW
        cv2.imshow(WINDOW_TITLE, myGUI.screen)

        # KEYBOARD COMMANDS
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        cv2.setMouseCallback(WINDOW_TITLE, myGUI.pointCrossMarker)
        if myGUI.save:
            cv2.imwrite('configs/calibration_image_markers.jpg', myGUI.screen)
            np.savetxt(f'configs/calibration_points2d.txt', points2d)
            np.savetxt(f'configs/calibration_points3d.txt', points3d)
            myGUI.save = False
        if quit:
            break
        else:
            myGUI.updateGUI(img)
        
        # SHOW POINTS POSITIONS
        for marker in myGUI.markers:
            [pixel, skip_marker] = marker
            pixel_x, pixel_y = pixel

            # BACK PROJECT PIXEL POSITION TO CAMERA 3D COORDINATES
            object_position = cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
            x, y, z = (position[0] for position in object_position)
            caption = f"Position:{x:.2f},{y:.2f}"
            myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+30)), 0.5)


