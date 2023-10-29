import os
import Calibration.interface as interface
import numpy as np
import cv2
import Vision.camera_transformation as camera_transformation

if __name__=="__main__":
    WINDOW_TITLE = "Camera Calibration"

    cwd = os.getcwd()

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = camera_transformation.Camera(
                camera_matrix=camera_matrix,
                camera_initial_position=calibration_position
                )
    PATH_TO_FIELD_POINTS = cwd+"/configs/field_points3d.txt"
    FIELD_POINTS = np.loadtxt(PATH_TO_FIELD_POINTS, dtype="float64")

    # VIDEO CAPTURE CONFIGS
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # IMAGE READ SETUP
    PATH_TO_IMG = cwd+"/configs/calibration_image.jpg"
    img = cv2.imread(PATH_TO_IMG)

    # USER INTERFACE SETUP
    myGUI = interface.GUI(
                        play = True,
                        mode = "calibration"
                        )

    while True:
        if cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               print("Check video capture path")
               break
           else: img = frame
        
        if myGUI.mode == "calibration":
            marking_done = myGUI.runUI(myGUI.screen)
            if marking_done==True:
                points2d = []
                points3d = []
                index = 0
                for marker in myGUI.markers:
                    [position,skip_marker] = marker
                    if not skip_marker:
                        points2d.append(position)
                        points3d.append(FIELD_POINTS[index])
                    index += 1
                points2d = np.array(points2d,dtype="float64")
                points3d = np.array(points3d,dtype="float64")
                camera_position, euler_angles = ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)


        # DISPLAY WINDOW
        cv2.imshow(WINDOW_TITLE,myGUI.screen)

        # KEYBOARD COMMANDS
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        cv2.setMouseCallback(WINDOW_TITLE, myGUI.pointCrossMarker)
        if myGUI.save:
            cv2.imwrite('configs/calibration_image.jpg', img)
            cv2.imwrite('configs/calibration_image_markers.jpg', myGUI.screen)
            np.savetxt(f'configs/calibration_position.txt', camera_position)
            np.savetxt(f'configs/calibration_rotation.txt', euler_angles)
            np.savetxt(f'configs/calibration_points2d.txt', points2d)
            np.savetxt(f'configs/calibration_points3d.txt', points3d)
            myGUI.save = False
            print(f'Camera Position:\n{camera_position}')
            print(f'Camera Rotation:\n{euler_angles}')
        if quit:
            break
        else:
            myGUI.updateGUI(img)
    
    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()
