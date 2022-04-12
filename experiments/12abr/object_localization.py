import numpy as np
import cv2
import math
import time

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
        points3d = self.fixPoints3d(points3d=points3d)
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
        #self.offset = (self.initial_position+camera_position.T).T

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

        p = s*leftsideMat+self.offset

        return p

    def ballAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.8):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def ballAsPointLinearRegression(self, left, top, right, bottom, weight_x, weight_y):
        x = [left, right, top, bottom, 1]@weight_x
        y = [left, right, top, bottom, 1]@weight_y
        return x, y
    
    def robotAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.9):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def goalAsPoint(self, left, top, right, bottom, weight_x = 0.5, weight_y=0.9):
        x = weight_x*left+(1-weight_x)*right
        y = weight_y*top+(1-weight_y)*bottom
        return x, y
    
    def cameraToPixelCoordinates(self, x, y, z_world=0):
        M = self.intrinsic_parameters
        R = self.rotation_matrix
        t = self.translation_vector
        height = self.height
        cameraPoint = np.array([(x,y,z_world-height)])
        offset = np.array([(self.offset_x,self.offset_y,0)])

        rightSideMat = M@(R@(cameraPoint+offset).T)

        s = rightSideMat[2]

        uvPoint = rightSideMat/s
        #import pdb; pdb.set_trace()
        return uvPoint

if __name__=="__main__":
    import object_detection
    import tensorrt as trt
    import interface
    
    # TEST FOR BALL LOCALIZATION ON IMAGE OR USB CAMERA CAPTURE
    test_x = 315
    test_y = 264

    # START TIME
    start = time.time()

    # SET WINDOW TITLE
    WINDOW_NAME = 'Object Localization'

    # USB CAMERA SETUP
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # IMAGE READ SETUP
    PATH_TO_IMG = r"/home/joao/ssl-detector/images/calibration_image_1.jpg"
    img = cv2.imread(PATH_TO_IMG)

    # OBJECT DETECTION MODEL
    trt_net = object_detection.DetectNet(
                model_path="/home/joao/ssl-detector/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt", 
                labels_path="/home/joao/ssl-detector/models/ssl_labels.txt", 
                input_width=300, 
                input_height=300,
                score_threshold = 0.32,
                draw = False,
                display_fps = False,
                TRT_LOGGER = trt.Logger(trt.Logger.INFO)
                )
    trt_net.loadModel()

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = "/home/joao/ssl-detector/configs/camera_matrix_C922.txt"
    PATH_TO_DISTORTION_PARAMETERS = "/home/joao/ssl-detector/configs/camera_distortion_C922.txt"
    PATH_TO_2D_POINTS = "/home/joao/ssl-detector/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = "/home/joao/ssl-detector/configs/calibration_points3d.txt"
    ssl_cam = Camera(
                camera_matrix_path=PATH_TO_INTRINSIC_PARAMETERS,
                camera_distortion_path=PATH_TO_DISTORTION_PARAMETERS
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # USER INTERFACE SETUP
    myGUI = interface.GUI(
                        screen = img.copy(),
                        play = True,
                        mode = "detection"
                        )
    
    while True:
        if cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               print("Check video capture path")
               break
           else: img = frame
        
        detections = trt_net.inference(img).detections

        for detection in detections:
            class_id, score, xmin, xmax, ymin, ymax = detection
            
            # BALL LOCALIZATION ON IMAGE
            if class_id==1:     # ball
                # COMPUTE PIXEL FOR BALL POSITION
                pixel_x, pixel_y = ssl_cam.ballAsPoint(left=xmin, top=ymin, right=xmax, bottom=ymax, weight_y = 0.25)

                # DRAW OBJECT POINT ON SCREEN
                myGUI.drawCrossMarker(myGUI.screen, int(pixel_x), int(pixel_y))

                # BACK PROJECT BALL POSITION TO CAMERA 3D COORDINATES
                object_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
                x, y, z = (position[0] for position in object_position)
                caption = f"Position:{x:.2f},{y:.2f}"
                myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+25)), 0.35)

        # DISPLAY WINDOW
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        run_time = time.time()-start
        myGUI.drawText(myGUI.screen, f"running time: {run_time:.2f}s", (8, 13), 0.5)
        cv2.imshow(WINDOW_NAME, myGUI.screen)
        
        if quit:
            break
        else: myGUI.updateGUI(img)
        
    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()