
import numpy as np
import cv2
import math

#img = cv2.imread('/home/joao/ssl-detector/6.jpg',1)
img = cv2.imread('../ssl-detector/25.jpg',1)
height = 0

test_point = np.array([(324,102,1)])
test_point = np.array([(321,246,1)])

# image 3.jpg 2d positions:
fieldCenter = (285,248)
leftCorner = (64,176)
rightCorner = (517, 184)

keeperAreaLeftTop = (181,177)
keeperAreaRightTop = (407,181)
keeperAreaLeftBottom = (161,186)
keeperAreaRightBottom = (432,190)

goalLeftTop = (255,161)
goalLeftBottom = (255,180)
goalRightTop = (337,158)
goalRightBottom = (337,179)
goalCenterBottom = (296,181)

#image 6.jpg
p1 = (40,107)   # left corner
p2 = (184,105)  # keeper left back
p3 = (274,103)  # goal left
p4 = (326,102)  # goal center
p5 = (374,100)  # goal right
p6 = (465,100)  # keeper right back
p7 = (610,99)   # right corner
p8 = (154,114)  # keeper left front
p9 = (505,109)  # keeper right front
p10 = (324,201) # field center

#image 23.jpg
#p1 = (73,241)   # left corner
p2 = (148,250)  # keeper left back
p3 = (260,247)  # goal left
p4 = (321,246)  # goal center
p5 = (383,245)  # goal right
p6 = (491,242)  # keeper right back
#p7 = (550,231)   # right corner
p8 = (94,265)  # keeper left front
p9 = (558,254)  # keeper right front
#p10 = (320,289) # field center


points2d = np.array([
                    p2,p3,p5,p6,p8,p9
                    ],dtype="float64")
#print("2d Points:",points2d)

# SSL Vision coordinates:
# ORIGIN: FIELD CENTER
# X AXIS: FRONT OF CAMERA VIEW TO BACK
# Y AXIS: LEFT OF CAMERA VIEW TO RIGHT
w1 = (-3000,-2000,0)
w2 = (-3000,-1000,0)
w3 = (-3000,-350,0)
w4 = (-3000,0,0)
w5 = (-3000,350,0)
w6 = (-3000,1000,0)
w7 = (-3000,2000,0)
w8 = (-2200,-1000,0)
w9 = (-2200,1000,0)
w10 = (0,0,0)

# Coordinates system parallel to camera vision axis:
# Center as origin
# y axis = center to goal
# x axis = left to right
dx = -2000
dy = +3000
w1 = (0+dx,0+dy,0)
w2 = (1000+dx,0+dy,0)
w3 = (1650+dx,0+dy,0)
w4 = (2000+dx,0+dy,0)
w5 = (2350+dx,0+dy,0)
w6 = (3000+dx,0+dy,0)
w7 = (4000+dx,0+dy,0)
w8 = (1000+dx,-800+dy,0)
w9 = (3000+dx,-800+dy,0)
w10 = (2000+dx,-3000+dy,0)

points3d = np.array([
                    w2,w3,w5,w6,w8,w9
                    ],dtype="float64")
#print("3D Points:",points3d)

mtx = np.array([
                (522.572,0,331.090),
                (0,524.896,244.689),
                (0,0,1)
                ])

dist=np.zeros((4,1))

# FIND CAMERA POSE
ret,rvec,tvec=cv2.solvePnP(points3d,points2d,mtx,dist)

#print("rvec")
#print(rvec)
#print("tvec")
#print(tvec)

rmtx, jacobian=cv2.Rodrigues(rvec)
print(math.degrees(rvec[0]), math.degrees(rvec[1]),math.degrees(rvec[2]))
print("Rotation Matrix")
print(rmtx)

pose = cv2.hconcat((rmtx,tvec))

K, R, _, rX, rY, rZ, euler_angles = cv2.decomposeProjectionMatrix(pose)

R1 = rZ.T @ rY.T @ rX.T
#print('Recombined Rotation Matrix')
#print(R1)

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

# FIND TEST POINT GLOBAL POSITION
leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(test_point)))
rightsideMat = np.matmul(np.linalg.inv(rmtx),tvec)

s = (height+rightsideMat[2])/leftsideMat[2]

p = np.matmul(np.linalg.inv(rmtx),(s*np.matmul(np.linalg.inv(mtx),np.transpose(test_point))-tvec))
p_x, p_y, p_z = float(p[0]),float(p[1]),float(p[2])

line_x = 320
line_y = 480
cv2.line(img, (line_x,0), (line_x,line_y), (0,0,0), 1)
cv2.circle(img,(int(test_point[0][0]),int(test_point[0][1])),3,(0,0,255),-1)
font = cv2.FONT_HERSHEY_SIMPLEX
'''cv2.putText(img,'pixel:' + str(int(test_point[0][0])) + ',' +
            str(int(test_point[0][1])), (int(test_point[0][0])+5,int(test_point[0][1])), font,
            0.5, (0,0,0), 1)
cv2.putText(img,f'field:{p_x:.2f},{p_y:.2f}',
            (int(test_point[0][0])+5,int(test_point[0][1])+15), font,
            0.5, (0,0,0), 1)'''

# FIND GLOBAL HEIGHT
print('CAMERA POSITION ON CALIBRATION IMAGE:')
cameraPosition = -np.matrix(rmtx).T*np.matrix(tvec)
print(cameraPosition)
camera_height = cameraPosition[2,0]

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

# GOAL CORNERS TO CAMERA POSITION
goalLeft = np.array([(259,247,1)])
cv2.circle(img,(int(goalLeft[0][0]),int(goalLeft[0][1])),3,(0,0,255),-1)
d1 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalLeft)))
s1 = -camera_height/d1[2]
p1 = s1*d1
print('LEFT GOAL CORNER POSITION')
print(p1)

goalRight = np.array([(381,245,1)])
cv2.circle(img,(int(goalRight[0][0]),int(goalRight[0][1])),3,(0,0,255),-1)
d2 = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(goalRight)))
s2 = -camera_height/d2[2]
p2 = s2*d2
print('RIGHT GOAL CORNER POSITION')
print(p2)

# ROBOT TO GOAL DISTANCES
d = cv2.sqrt(pCam[0]*pCam[0]+pCam[1]*pCam[1])
print(f'Goal Center absolute distance: {d}')

d1 = cv2.sqrt(p1[0]*p1[0]+p1[1]*p1[1])
print(f'Goal Left absolute distance: {d1}')

d2 = cv2.sqrt(p2[0]*p2[0]+p2[1]*p2[1])
print(f'Goal Right absolute distance: {d2}')

# ROBOT TO GOAL ROTATION
l = 710            # goal length in milimeters
#l = p2[0]-p1[0]     # goal length according to camera positions

cos1 = (d*d + l*l/4 - d1*d1)/(d*l)
theta1 = math.degrees(np.arccos(cos1))
print(f'Goal Left rotation cosine: {cos1}, {theta1}')

cos2 = (d*d + l*l/4 - d2*d2)/(d*l)
theta2 = math.degrees(np.arccos(cos2))
print(f'Goal Right rotation cosine: {cos2}, {theta2}')

theta = 90-theta1
print(f'Robot to field rotation in degrees: {theta}')

cos_phi1 = (l*l+d1*d1-d2*d2)/(2*d1*l)
cos_phi2 = (l*l+d2*d2-d1*d1)/(2*d2*l)
l1 = d1*cos_phi1
l2 = d2*cos_phi2
print(f'phi1 cosine: {cos_phi1}')
print(f'phi2 cosine: {cos_phi2}')
print(f'Goal length={l1}+{l2}={l1+l2}')

# ROBOT TO ABSOLUTE POSITION FROM THETA 1
theta = math.radians(90-theta1)
dx = d*math.cos(theta)
dy = d*math.sin(theta)
x = dx-3000
y = -dy
print(f'Robot to field absolute position from theta1: {x}, {y}')

# ROBOT TO ABSOLUTE POSITION FROM THETA 2
theta = math.radians(theta2-90)
dx = d*math.cos(theta)
dy = d*math.sin(theta)
x = dx-3000
y = -dy
print(f'Robot to field absolute position from theta2: {x}, {y}')

cv2.imshow('ssl', img)

while True:
    key=cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break