import numpy as np
import cv2
import interface
import object_localization
import os

if __name__=="__main__":
    test_point = [0, 1500]
    offset = [21.32-0, 491.80-500]
    test_point = np.add(test_point,offset)

    WINDOW_TITLE = "test"
    cwd = os.getcwd()
    img = cv2.imread(cwd+'/experiments/9abr/1.jpg')
    img = cv2.resize(img, (640, 480))

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/camera_matrix_C922.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/camera_distortion_C922.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    ssl_cam = object_localization.Camera(
                camera_matrix_path=PATH_TO_INTRINSIC_PARAMETERS,
                camera_distortion_path=PATH_TO_DISTORTION_PARAMETERS
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    # USER INTERFACE SETUP
    myGUI = interface.GUI(
                        screen = img.copy(),
                        mode = "debug"
                        )

    while True:
        
        myGUI.runUI(myGUI.screen)
        world_x, world_y = test_point
        uvPoint = ssl_cam.cameraToPixelCoordinates(world_x, world_y, z_world=0)
        pixel_x, pixel_y = int(uvPoint[0]), int(uvPoint[1])

        if myGUI.mode == 'debug':
            myGUI.current_marker = [(pixel_x, pixel_y), False]

        cv2.imshow(WINDOW_TITLE,myGUI.screen)

        # KEYBOARD COMMANDS
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        if quit:
            break

        cv2.setMouseCallback(WINDOW_TITLE, myGUI.pointCrossMarker)
        myGUI.updateGUI(img)
    
    cv2.destroyAllWindows()