import cv2
import numpy as np

img = cv2.imread('grid-cup1.jpg',1)
height = 0

#test_point = np.array([(368,425,1)])
test_point = np.array([(219,291,1)])
test_point = np.array([(624,240,1)])

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
print("Camera Matrix")
if np.shape(mtx)==(3,3): print(mtx)

dist=np.zeros((4,1))
print("Distortion")
print(dist)

ret,rvec,tvec=cv2.solvePnP(points3d,points2d,mtx,dist)

print("rvec")
print(rvec)
print("tvec")
print(tvec)

rmtx, jacobian=cv2.Rodrigues(rvec)
print("Rotation Matrix")
print(rmtx)

# test reprojection
print("Test Point")
print(np.transpose(test_point))

leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(test_point)))
rightsideMat = np.matmul(np.linalg.inv(rmtx),tvec)

print("Right Side Matrix")
print(rightsideMat)
print("Left Side Matrix")
print(leftsideMat)

s = (height+rightsideMat[2])/leftsideMat[2]
print("s =",s)

p = np.matmul(np.linalg.inv(rmtx),(s*np.matmul(np.linalg.inv(mtx),np.transpose(test_point))-tvec))
print("p =")
print(p)
p_x, p_y, p_z = float(p[0]),float(p[1]),float(p[2])
print(f'X = {p_x:.2f}')
print(f'Y = {p_y:.2f}')
print(f'Z = {p_z:.2f}')

print('CAMERA POSITION:')
cameraPosition = -np.matrix(rmtx).T*np.matrix(tvec)
print(cameraPosition)