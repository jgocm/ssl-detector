import cv2
import numpy as np
import math

#img = cv2.imread('/home/joao/ssl-detector/experiments/25.jpg')
img = cv2.imread('experiments/30.jpg')
#img = cv2.imread('30.jpg')

nr = 30
points2d = np.loadtxt(f'experiments/{nr}_points2d.txt', dtype="float64")
points3d = np.loadtxt(f'experiments/{nr}_points3d.txt', dtype="float64")

mtx = np.array([
                (522.572,0,331.090),
                (0,524.896,244.689),
                (0,0,1)
                ])

dist=np.zeros((4,1))

# FIND CAMERA CALIBRATION POSE
ret,rvec,tvec=cv2.solvePnP(points3d,points2d,mtx,dist)
rmtx, jacobian=cv2.Rodrigues(rvec)

pose = cv2.hconcat((rmtx,tvec))

K, R, _, rX, rY, rZ, euler_angles = cv2.decomposeProjectionMatrix(pose)

theta_x = euler_angles[0][0]    # global x axis
theta_y = euler_angles[1][0]    # global y axis
theta_z = euler_angles[2][0]    # global z axis
print(f'Rotation angles: {theta_x:.4f}, {theta_y:.4f}, {theta_z:.4f}')

theta_x = math.radians(theta_x)
cx = math.cos(theta_x)
sx = math.sin(theta_x)
rX_cam = np.array([
    [1,0,0],
    [0,cx,-sx],
    [0,sx,cx]
])

#theta_y = 0
theta_y = math.radians(theta_y)
cy = math.cos(theta_y)
sy = math.sin(theta_y)
rY_cam = np.array([
    [cy,0,sy],
    [0,1,0],
    [-sy,0,cy]
])

#theta_z = 0
theta_z = math.radians(theta_z)
cz = math.cos(theta_z)
sz = math.sin(theta_z)
rZ_cam = np.array([
    [cz,-sz,0],
    [sz,cz,0],
    [0,0,1]
])

R2 = rZ_cam @ rY_cam @ rX_cam

print('New Recombined Rotation Matrix')
print(R2)
#rmtx = R2

# CAMERA HEIGHT
print('CAMERA ABSOLUTE POSITION ON CALIBRATION IMAGE:')
cameraPosition = -np.matrix(rmtx).T*np.matrix(tvec)
print(cameraPosition)
camera_height = cameraPosition[2,0]

# TEST POINT:
test_point = np.array([(318,52,1)])     # goal middle
test_point_height = 0

# FIND TEST POINT GLOBAL POSITION
leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(test_point)))
rightsideMat = np.matmul(np.linalg.inv(rmtx),tvec)
s = (test_point_height+rightsideMat[2])/leftsideMat[2]

p = np.matmul(np.linalg.inv(rmtx),(s*np.matmul(np.linalg.inv(mtx),np.transpose(test_point))-tvec))
#p = s*leftsideMat-rightsideMat
p_x, p_y, p_z = float(p[0]),float(p[1]),float(p[2])
#print(f's={s}')
print('GOAL CENTER ABSOLUTE POSITION ON CALIBRATION IMAGE:')
#print(f'p: {p_x:.2f}, {p_y:.2f}, {p_z:.2f}')
print(p)

# POINT TO CAMERA RELATIVE POSITION
s = camera_height/leftsideMat[2]

pCam = -s*leftsideMat
print('POINT TO CAMERA RELATIVE POSITION WITH ORIGINAL ROTATION MATRIX')
print(pCam)

# POINT TO CAMERA RELATIVE POSITION WITH NEW ROTATION MATRIX
rmtx = R2
leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(test_point)))
s = camera_height/leftsideMat[2]
pCam = -s*leftsideMat
print('POINT TO CAMERA RELATIVE POSITION WITH NEW ROTATION MATRIX')
print(pCam)

# GOAL POINTS RELATIVE POSITIONS TO CAMERA
goalMiddle = np.array([(318,52,1)])
cv2.circle(img,(int(goalMiddle[0][0]),int(goalMiddle[0][1])),3,(0,0,255),-1)
d = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalMiddle)))
s = camera_height/d[2]
p = -s*d
x, y = p[0], p[1]
print('GOAL CENTER RELATIVE POSITION')
print(p)

goalLeft = np.array([(252,52,1)])
cv2.circle(img,(int(goalLeft[0][0]),int(goalLeft[0][1])),3,(0,0,255),-1)
d1 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalLeft)))
s1 = camera_height/d1[2]
p1 = -s1*d1
x1, y1 = p1[0], p1[1]
print('GOAL LEFT CORNER RELATIVE POSITION')
print(p1)

goalRight = np.array([(383,52,1)])
cv2.circle(img,(int(goalRight[0][0]),int(goalRight[0][1])),3,(0,0,255),-1)
d2 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalRight)))
s2 = camera_height/d2[2]
p2 = -s2*d2
x2, y2 = p2[0],p2[1]
print('GOAL RIGHT CORNER RELATIVE POSITION')
print(p2)

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

print('CAMERA GLOBAL POSITION DURING CALIBRATION')
print(rightsideMat)

# GOAL LENGTH
#l = 710                    # goal length in milimeters  
l = c*(x2-x1)-s*(y2-y1)     # goal length according to camera positions
print(f'goal length={l}')

while True:
    cv2.imshow('img', img)
    key = cv2.waitKey(0) 
    if key & 0xFF==ord('q'):
        break