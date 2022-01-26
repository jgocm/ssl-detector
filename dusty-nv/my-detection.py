import os
import sys
import cv2
import numpy as np

import jetson.inference
import jetson.utils

def locate_goal(top, left, bottom, right, mtx, dist):
	points2d = np.array([
                    (top,left),
                    (top,right),
                    (bottom,left),
                    (bottom,right)],
					dtype="float64")

	points3d = np.array([
                    (0,0,50),
                    (0,400,50),
                    (0,0,0),
                    (0,400,0)],
					dtype="float64")
	
	ret,rvec,tvec=cv2.solvePnP(points3d,points2d,mtx,dist)

	return rvec, tvec

def locate_ball(center, mtx, rvec, tvec):
	rmtx, jacobian=cv2.Rodrigues(rvec)

	leftsideMat = np.matmul(np.linalg.inv(rmtx),np.matmul(np.linalg.inv(mtx),np.transpose(center)))
	rightsideMat = np.matmul(np.linalg.inv(rmtx),tvec)

	ball_radius=21.335

	s = (ball_radius+rightsideMat[2])/leftsideMat[2]
	p = np.matmul(np.linalg.inv(rmtx),(s*np.matmul(np.linalg.inv(mtx),np.transpose(center))-tvec))

	p_x, p_y, p_z = float(p[0]),float(p[1]),float(p[2])
	print(f'X = {p_x:.2f}')
	print(f'Y = {p_y:.2f}')
	print(f'Z = {p_z:.2f}')


# camera parameters
mtx = np.array([
                (522.572,0,331.090),
                (0,524.896,244.689),
                (0,0,1)
                ])
dist=np.zeros((4,1))
# init rotation and translation vectors
rvec=np.zeros((3,1))
tvec=np.zeros((3,1))
ball_center=np.array([(0,0,1)])


# network args
path = os.path.dirname(os.getcwd()) #parent folder path
ssl_mb2_model='--model=/home/joao/ssl-detector/models/ssl3/mb2-ssd-lite.onnx'
ssl_labels='--labels=/home/joao/ssl-detector/models/labels.txt'
batch_size ='--batch-size=4'
input_blob='--input-blob=input_0'
output_cvg='--output-cvg=scores'
output_bbox='--output-bbox=boxes'

net_args = sys.argv
net_args.extend([ssl_mb2_model,
				ssl_labels,
				batch_size,
				input_blob,
				output_cvg,
				output_bbox
				])
net = jetson.inference.detectNet("ssd-mobilenet-v2", net_args, threshold=0.5)

# OpenCV configs
font = cv2.FONT_HERSHEY_SIMPLEX
# display dimensions
dispW = 640
dispH = 480
camera = cv2.VideoCapture('/dev/video0')
camera.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

image=cv2.imread('/home/joao/ssl-dataset/1_resized/00285.jpg')

while True:
	#_, img = camera.read()
	img=image
	height = img.shape[0]
	width = img.shape[1]

	frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)
	frame = jetson.utils.cudaFromNumpy(frame)

	detections = net.Detect(frame, width, height)

	#display.RenderOnce(img, width, height)
	#display.SetStatus("Object Detection - Network {:.0f} FPS".format(net.GetNetworkFPS()))
	#img = jetson.utils.cudaToNumpy(img)
	#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	for detection in detections:
		print(detection)
		ID = detection.ClassID			# class ID number
		item = net.GetClassDesc(ID)		# class ID title
		confidence = detection.Confidence
		top = int(detection.Top)		# top position
		bottom = int(detection.Bottom)	# bottom position
		left = int(detection.Left)		# left position
		right = int(detection.Right)	# right position
		cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 1)
		cv2.putText(img, item, (left,top+20), font, .75, (0,0,255), 2)

		if item == 'goal':
			rvec, tvec = locate_goal(top=top, left=left, bottom=bottom, right=right, mtx=mtx, dist=dist)
			print("Rotation Vector:", rvec)
			print("Translation Vector:", tvec)

		if item == 'ball':
			ball_center[0,0]=detection.Center[0]
			ball_center[0,1]=detection.Center[1]
			locate_ball(center=ball_center, mtx=mtx, rvec=rvec, tvec=tvec)
	
	cv2.imshow('detCam', img)
	if cv2.waitKey(1)==ord('q'):
		break

camera.release()
cv2.destroyAllWindows()