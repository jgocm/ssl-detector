import numpy as np
import cv2
import interface
import object_localization
import os

if __name__=="__main__":
    test_point = [0, 500]

    WINDOW_TITLE = "test"
    cwd = os.getcwd()
    img = cv2.imread(cwd+'/experiments/12abr/1.jpg')
    img = cv2.resize(img, (640, 480))

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    camera_distortion = np.loadtxt(PATH_TO_DISTORTION_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = object_localization.Camera(
                camera_matrix=camera_matrix,
                #camera_distortion=camera_distortion,
                #camera_initial_position=calibration_position
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)
    print(ssl_cam.position)

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