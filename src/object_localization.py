import cv2
import numpy as np
import math

class camera():
    def __init__(
                self,
                camera_path="/dev/video0",
                camera_matrix=np.identity(3),
                camera_distortion=np.zeros((4,1)),
                camera_height=0,
                camera_initial_position=None
                ):
        super(camera, self).__init__()
        self.path = camera_path
        self.intrinsic_parameters = camera_matrix
        self.distortion_parameters = camera_distortion

        euler_angles = np.array([(94.0588, -3.8788, -1.2585)]).T

        self.rotation_vector = None
        self.rotation_matrix = None
        self.translation_vector = None

        self.calib_position = None
        self.calib_rotation = euler_angles
        self.calib_height = camera_height

        self.position = camera_initial_position
        self.rotation = euler_angles
        self.height = camera_height
        
    def getIntrinsicParameters(self):
        if np.shape(self.intrinsic_parameters)==(3,3):
            #print(f"Camera Matrix is:")
            #print(self.intrinsic_parameters)
            return self.intrinsic_parameters
        else:
            #print("Camera parameters are not defined")
            return 0
    
    def getDistortionParameters(self):
        return self.distortion_parameters
    
    def getPose(self):
        if np.shape(self.rotation_vector)==(3,1):
            if np.shape(self.translation_vector)==(3,1):
                #print("Rotation Vector is:")
                #print(self.rotation_vector)
                #print("Translation Vector is:")
                #print(self.translation_vector)
                return self.rotation_matrix, self.translation_vector
            else:
                #print("Translation Vector is not defined")
                return 0
        else:
            #print("Rotation Vector is not defined")
            return 0
    
    def getRotationVector(self):
        if np.shape(self.rotation_vector)==(3,1):
            #print("Rotation Vector is:")
            #print(self.rotation_vector)
            return self.rotation_vector
        else:
            #print("Rotation Vector is not defined")
            return 0
    
    def getTranslationVector(self):
        if np.shape(self.translation_vector)==(3,1):
            #print("Translation Vector is:")
            #print(self.rotation_vector)
            return self.rotation_vector
        else:
            #print("Translation Vector is not defined")
            return 0

    def setIntrinsicParameters(self, mtx):
        if np.shape(mtx)==(3,3): 
            self.intrinsic_parameters = mtx
            #print(f"Camera Matrix is:")
            #print(mtx)
            return self
        else:
            #print(f"Camera Matrix must be of shape (3,3) and the inserted matrix has shape {np.shape(mtx)}")
            return self
    
    def setDistortionParameters(self, dist):
        if np.shape(dist)==(4,1):
            self.distortion_parameters = dist
            #print("Camera distortion parameters are:")
            #print(dist)
            return True
        else:
            #print(f"Camera distortion parameters must be of shape (4,1) and the inserted array has shape {np.shape(dist)}")
            return False
    
    def setPose(self, points3d, points2d):
        _,rvec,tvec=cv2.solvePnP(points3d,points2d,self.intrinsic_parameters,self.distortion_parameters)
        self.setRotationVector(rvec)
        self.setRotationMatrix(rvec)
        self.setTranslationVector(tvec)
        
        pose = cv2.hconcat((self.rotation_matrix,tvec))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose)

        camera_position = -np.matrix(self.rotation_matrix).T*np.matrix(tvec)
        height = camera_position[2,0]

        self.calib_position = camera_position
        self.calib_rotation = euler_angles
        self.calib_height = height

        self.position = camera_position
        self.rotation = euler_angles
        self.camera_height = height

        return camera_position, euler_angles, height

    def setRotationVector(self, rvec):
        if np.shape(rvec)==(3,1):
            self.rotation_vector = rvec
            #print("Rotation Vector is:")
            #print(self.rotation_vector)
            return True
        else:
            #print(f"Camera rotation vector must be of shape (3,1) and the inserted array has shape {np.shape(rvec)}")
            return False

    def setRotationMatrix(self, rvec):
        rmtx, jacobian=cv2.Rodrigues(rvec)
        self.rotation_matrix=rmtx

    def setTranslationVector(self, tvec):
        if np.shape(tvec)==(3,1):
            self.translation_vector = tvec
            #print("Translation Vector is:")
            #print(self.translation_vector)
            return True
        else:
            #print(f"Camera translation vector must be of shape (3,1) and the inserted array has shape {np.shape(tvec)}")
            return False    
    
    def setCameraHeight(self, height):
        self.camera_height = height
    
    def pixelToWorldCoordinates(self, x, y, height):
        uvPoint = np.array([(x, y, 1)])
        tvec = self.translation_vector
        rmtx = self.rotation_matrix
        mtx = self.intrinsic_parameters

        leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(uvPoint)))
        rightsideMat = np.matmul(np.linalg.inv(rmtx),tvec)

        s = (height+rightsideMat[2])/leftsideMat[2]

        p = np.matmul(np.linalg.inv(rmtx),(s*np.matmul(np.linalg.inv(mtx),np.transpose(uvPoint))-tvec))

        return p

    def pixelToCameraCoordinates(self, x, y):
        uvPoint = np.array([(x,y,1)])
        mtx = self.intrinsic_parameters
        euler_angles = self.rotation
        height = self.height
        
        # COMPUTE RELATIVE ROTATION MATRIX
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
        #theta_y = 0
        theta_y = math.radians(theta_y)
        cy = math.cos(theta_y)
        sy = math.sin(theta_y)
        rY_cam = np.array([
            [cy,0,sy],
            [0,1,0],
            [-sy,0,cy]
        ])

        theta_z = euler_angles[2][0]
        #theta_z = 0
        theta_z = math.radians(theta_z)
        cz = math.cos(theta_z)
        sz = math.sin(theta_z)
        rZ_cam = np.array([
            [cz,-sz,0],
            [sz,cz,0],
            [0,0,1]
        ])
        
        rmtx = rZ_cam @ rY_cam @ rX_cam
        leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(uvPoint)))
        s = -height/leftsideMat[2]

        p = s*leftsideMat

        return p

    def updateRotation(self, theta_z):
        '''
        Angles must be in degrees
        '''
        euler_angles = self.rotation

        # COMPUTE NEW ROTATION MATRIX
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

        #theta_z = euler_angles[2][0]
        theta_z = math.radians(theta_z)
        cz = math.cos(theta_z)
        sz = math.sin(theta_z)
        rZ_cam = np.array([
            [cz,-sz,0],
            [sz,cz,0],
            [0,0,1]
        ])

        rmtx = rZ_cam @ rY_cam @ rX_cam
        self.rotation_matrix = rmtx
        self.rotation = np.array([(theta_x, theta_y, theta_z)]).T

    def computePoseToBBox(self, left, top, right, bottom):
        rmtx = self.rotation_matrix
        mtx = self.intrinsic_parameters
        camera_height = self.height

        objTop = 0.9*top+0.1*bottom
        objBottom = 0.1*top+0.9*bottom
        objLeft = 0.85*left+0.15*right
        objRight = 0.15*left+0.85*right

        # BOUNDING BOX CORNERS TO CAMERA RELATIVE POSITIONS
        x,y = (left+right)/2, objBottom                    # goal lower center
        x1,y1 = objLeft, objBottom                         # goal lower left corner
        x2,y2 = objRight, objBottom                        # goal lower right corner

        p = self.pixelToCameraCoordinates(x,y)      # goal lower center relative position to camera
        p1 = self.pixelToCameraCoordinates(x1,y1)   # goal lower left corner relative position to camera
        p2 = self.pixelToCameraCoordinates(x2,y2)   # goal lower right corner relative position to camera

        # AXIS ROTATION
        tan_theta = (y1-y2)/(x2-x1)
        theta=np.arctan(tan_theta)
        #theta=-0.7629
        print(f'Z AXIS ROTATION IN DEGREES:')
        print(f'theta={math.degrees(theta)}')
        #theta = math.radians(theta)
        s = math.sin(theta)
        c = math.cos(theta)
        print(f'theta sine={s}')
        print(f'theta cosine={c}')

        # ROBOT ABSOLUTE POSITION
        x0 = 0
        y0 = 3000
        xt = c*x-s*y-x0
        yt = s*x+c*y-y0
        print('ROBOT TRANSLATION BASED ON GOAL CENTER')
        print(xt,yt)

        x0 = -350
        y0 = 3000
        xt = c*x1-s*y1-x0
        yt = s*x1+c*y1-y0
        print('ROBOT TRANSLATION BASED ON GOAL LEFT CORNER')
        print(xt,yt)

        x0 = 350
        y0 = 3000
        xt = c*x2-s*y2-x0
        yt = s*x2+c*y2-y0
        print('CAMERA TRANSLATION BASED ON GOAL RIGHT CORNER')
        print(xt,yt)

        # GOAL LENGTH
        #l = 710                    # goal length in milimeters  
        l = c*(x2-x1)-s*(y2-y1)     # goal length according to camera positions
        print(f'goal length={l}')
        #return position
              

def main():
    print("Test")
    #height = 214.60
    height = 320.00

    #test_point = np.array([(368,425,1)])
    test_point = (219,291,1)
    x = 219
    y = 291

    points2d = np.array([
                        (624,240),
                        (532,435),
                        (82,335),
                        (467,238),
                        (432,235),
                        (529,289)
                        ],dtype="float64")
    print("2d Points:",points2d)

    points3d = np.array([
                        (0,0,0),
                        (0,1100,0),
                        (600,1100,0),
                        (400,0,0),
                        (500,0,0),
                        (100,600,0)
                        ],dtype="float64")
    print("3D Points:",points3d)

    mtx = np.array([
                    (522.572,0,331.090),
                    (0,524.896,244.689),
                    (0,0,1)
                    ])

    myCam = camera(camera_path="/dev/video0",
                    camera_matrix=mtx,
                    camera_height=height
                    )
                    
    myCam.setCameraHeight(height)
    myCam.setPose(points3d, points2d)

    print(myCam.pixelToCameraCoordinates(x, y))

    print(myCam.pixelToWorldCoordinates(x,y,0))

if __name__=="__main__":
    main()