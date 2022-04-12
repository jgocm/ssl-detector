import os
import numpy as np
import cv2
import math
import object_detection
import tensorrt as trt
import interface
import time
import object_localization

if __name__=="__main__":
    cwd = os.getcwd()

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
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/camera_matrix_C922.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/camera_distortion_C922.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    camera_distortion = np.loadtxt(PATH_TO_DISTORTION_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = object_localization.Camera(
                camera_matrix=camera_matrix,
                camera_distortion=camera_distortion,
                camera_initial_position=calibration_position
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

    # EXPERIMENT SAVING CONFIGS
    PATH_TO_EXPERIMENT = "/home/joao/ssl-detector/experiments/12abr"
    np.savetxt(f'{PATH_TO_EXPERIMENT}/calibration_points2d.txt', points2d)
    np.savetxt(f'{PATH_TO_EXPERIMENT}/calibration_points3d.txt', points3d)
    cv2.imwrite(f'{PATH_TO_EXPERIMENT}/calibration_image.jpg', img)
    nr=1
    ball_detected = False

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

                # SAVE BOUNDING BOX COORDINATES
                ball_detected = True
                bbox = [xmin, xmax, ymin, ymax]

            else: ball_detected = False
                
        # DISPLAY WINDOW
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        run_time = time.time()-start
        myGUI.drawText(myGUI.screen, f"running time: {run_time:.2f}s", (8, 13), 0.5)
        cv2.imshow(WINDOW_NAME, myGUI.screen)
        if key == ord('s'):
            np.savetxt(f'{PATH_TO_EXPERIMENT}/{nr}.txt', bbox)
            cv2.imwrite(f'{PATH_TO_EXPERIMENT}/{nr}.jpg', img)
            nr+=1
        elif quit:
            break
        else: myGUI.updateGUI(img)
        
    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()