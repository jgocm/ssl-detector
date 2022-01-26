import cv2
import numpy as np
import math

img = cv2.imread('/home/joao/ssl-detector/experiments/25.jpg')

nr = 25
points2d = np.loadtxt(f'experiments/{nr}_points2d.txt', dtype="float64")
points3d = np.loadtxt(f'experiments/{nr}_points3d.txt', dtype="float64")

mtx = np.array([
                (522.572,0,331.090),
                (0,524.896,244.689),
                (0,0,1)
                ])

dist=np.zeros((4,1))

# FIND CAMERA POSE
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
print('CAMERA POSITION ON CALIBRATION IMAGE:')
cameraPosition = -np.matrix(rmtx).T*np.matrix(tvec)
print(cameraPosition)
camera_height = cameraPosition[2,0]

# TEST POINT:
test_point = np.array([(320,51,1)])

# FIND TEST POINT GLOBAL POSITION
leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(test_point)))
rightsideMat = np.matmul(np.linalg.inv(rmtx),tvec)

s = (camera_height+rightsideMat[2])/leftsideMat[2]

p = -np.matmul(np.linalg.inv(rmtx),(s*np.matmul(np.linalg.inv(mtx),np.transpose(test_point))-tvec))
p_x, p_y, p_z = float(p[0]),float(p[1]),float(p[2])
print(f'p: {p_x:.2f}, {p_y:.2f}, {p_z:.2f}')

# POINT TO CAMERA RELATIVE POSITION
s = -camera_height/leftsideMat[2]

pCam = s*leftsideMat
print('POINT TO CAMERA RELATIVE POSITION WITH ORIGINAL ROTATION MATRIX')
print(pCam)

# POINT TO CAMERA RELATIVE POSITION WITH NEW ROTATION MATRIX
rmtx = R2
leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(test_point)))
s = -camera_height/leftsideMat[2]
pCam = s*leftsideMat
print('POINT TO CAMERA RELATIVE POSITION WITH NEW ROTATION MATRIX')
print(pCam)

# GOAL POINTS TO CAMERA POSITION
goalMiddle = np.array([(320,52,1)])
cv2.circle(img,(int(goalMiddle[0][0]),int(goalMiddle[0][1])),3,(0,0,255),-1)
d = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalMiddle)))
s = -camera_height/d[2]
p = s*d
x, y = p[0], p[1]
print('GOAL MIDDLE POSITION')
print(x, y)

goalLeft = np.array([(255,52,1)])
cv2.circle(img,(int(goalLeft[0][0]),int(goalLeft[0][1])),3,(0,0,255),-1)
d1 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalLeft)))
s1 = -camera_height/d1[2]
p1 = s1*d1
x1, y1 = p1[0], p1[1]
print('GOAL LEFT CORNER POSITION')
print(x1, y1)

goalRight = np.array([(385,52,1)])
cv2.circle(img,(int(goalRight[0][0]),int(goalRight[0][1])),3,(0,0,255),-1)
d2 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalRight)))
s2 = -camera_height/d2[2]
p2 = s2*d2
x2, y2 = p2[0],p2[1]
print('GOAL RIGHT CORNER POSITION')
print(x2, y2)

# ROBOT TO GOAL DISTANCES
d = cv2.sqrt(p[0]*p[0]+p[1]*p[1])
print(f'Goal Center absolute distance: {d}')

d1 = cv2.sqrt(p1[0]*p1[0]+p1[1]*p1[1])
print(f'Goal Left absolute distance: {d1}')

d2 = cv2.sqrt(p2[0]*p2[0]+p2[1]*p2[1])
print(f'Goal Right absolute distance: {d2}')

# AXIS ROTATION
tan_theta = (y1-y2)/(x2-x1)
theta=np.arctan(tan_theta)
print(f'theta={theta}')
s = math.sin(theta)
c = math.cos(theta)

# GOAL LENGTH
#l = 710                    # goal length in milimeters  
l = c*(x2-x1)-s*(y2-y1)     # goal length according to camera positions
print(f'goal length={l}')

# GOAL ABSOLUTE POSITION
x0 = s*y-c*x
y0 = 3000-s*x-c*y
print(x0,y0)

while True:
    cv2.imshow('img', img)
    key = cv2.waitKey(0) 
    if key & 0xFF==ord('q'):
        break

