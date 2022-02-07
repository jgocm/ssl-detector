import cv2
import numpy as np
import math

#img = cv2.imread('/home/joao/ssl-detector/experiments/25.jpg')
#img = cv2.imread('experiments/30.jpg')
img = cv2.imread('70.jpg')

nr = 71
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

#euler_angles = np.loadtxt(f'experiments/{nr}_camera_rotation.txt', dtype="float64")

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

# TEST POINTS:
'''
IMAGE 42:
goalCenter = np.array([(318,79,1)])
goalLeft = np.array([(198,85,1)])
goalRight = np.array([(406,75,1)])'''

'''
IMAGE 39:
goalCenter = np.array([(322,56,1)])
goalLeft = np.array([(256,56,1)])
goalRight = np.array([(388,56,1)])'''

'''IMAGE 58:
goalCenter = np.array([(305,70,1)])
goalLeft = np.array([(219,78,1)])
goalRight = np.array([(372,65,1)])'''

'''IMAGE 51:
goalCenter = np.array([(313,51,1)])
goalLeft = np.array([(246,51,1)])
goalRight = np.array([(377,51,1)])'''

'''IMAGE 47:
goalCenter = np.array([(322,52,1)])
goalLeft = np.array([(258,52,1)])
goalRight = np.array([(389,52,1)])'''

'''
#IMAGE 62
goalLeft = np.array([(293,78,1)])
goalCenter = np.array([(400,78,1)])
goalRight = np.array([(507,78,1)])'''

goalLeft = np.array([(262,50,1)])
goalCenter = np.array([(327,50,1)])
goalRight = np.array([(393,50,1)])

test_point = goalCenter     # goal middle
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
print(pCam+cameraPosition)

# GOAL POINTS RELATIVE POSITIONS TO CAMERA
goalMiddle = goalCenter
cv2.circle(img,(int(goalMiddle[0][0]),int(goalMiddle[0][1])),3,(0,0,255),-1)
d = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalMiddle)))
s = camera_height/d[2]
p = -s*d+cameraPosition
x, y = p[0], p[1]
print('GOAL CENTER RELATIVE POSITION')
print(p)

cv2.circle(img,(int(goalLeft[0][0]),int(goalLeft[0][1])),3,(0,0,255),-1)
d1 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalLeft)))
s1 = camera_height/d1[2]
p1 = -s1*d1+cameraPosition
x1, y1 = p1[0], p1[1]
print('GOAL LEFT CORNER RELATIVE POSITION')
print(p1)

cv2.circle(img,(int(goalRight[0][0]),int(goalRight[0][1])),3,(0,0,255),-1)
d2 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalRight)))
s2 = camera_height/d2[2]
p2 = -s2*d2+cameraPosition
x2, y2 = p2[0],p2[1]
print('GOAL RIGHT CORNER RELATIVE POSITION')
print(p2)

# AXIS ROTATION
tan_theta = (y1-y2)/(x2-x1)
theta=np.arctan(tan_theta)

#theta=-2*np.arctan(np.sqrt(x2-x1-355)/np.sqrt(x2-x1+355))
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
print(-xt,-yt)

x0 = -350
y0 = 3000
xt = c*x1-s*y1-x0
yt = s*x1+c*y1-y0
print('ROBOT TRANSLATION BASED ON GOAL LEFT CORNER')
print(-xt,-yt)

x0 = 350
y0 = 3000
xt = c*x2-s*y2-x0
yt = s*x2+c*y2-y0
print('CAMERA TRANSLATION BASED ON GOAL RIGHT CORNER')
print(-xt,-yt)

# GOAL LENGTH
L = 710                     # goal length in milimeters  
l = c*(x2-x1)-s*(y2-y1)     # goal length according to camera positions
print(f'goal length={l}')

# error measure:
e = L - l

while True:
    cv2.imshow('img', img)
    key = cv2.waitKey(1) 
    if key & 0xFF==ord('q'):
        break