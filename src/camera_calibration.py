import os
import interface
import numpy as np
import cv2
import object_localization

if __name__=="__main__":
    WINDOW_TITLE = "Camera Calibration"

    cwd = os.getcwd()

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    PATH_TO_FIELD_POINTS = cwd+"/configs/arena_points3d.txt"
    FIELD_POINTS = np.loadtxt(PATH_TO_FIELD_POINTS, dtype="float64")
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    cam = object_localization.Camera(camera_matrix=camera_matrix)

    # IMAGE READ SETUP
    PATH_TO_IMG = cwd+"/configs/calibration_image.jpg"
    img = cv2.imread(PATH_TO_IMG)

    # USER INTERFACE SETUP
    myGUI = interface.GUI(play = True,
                          mode = "calibration")

    while True:        
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
                camera_position, euler_angles = cam.computePoseFromPoints(points3d=points3d, points2d=points2d)
        else:
            marking_done = myGUI.runUI(myGUI.screen)

        # DISPLAY WINDOW
        cv2.imshow(WINDOW_TITLE,myGUI.screen)

        # KEYBOARD COMMANDS
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        cv2.setMouseCallback(WINDOW_TITLE, myGUI.pointCrossMarker)
        if myGUI.save:
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
    cv2.destroyAllWindows()
