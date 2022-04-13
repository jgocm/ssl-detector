import numpy as np
import cv2
import interface
import object_localization
import os

if __name__=="__main__":
    WINDOW_TITLE = "test"

    cwd = os.getcwd()

    img = cv2.imread(cwd+'/configs/calibration_image.jpg')
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
                camera_initial_position=calibration_position
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
        
        if myGUI.mode == 'debug':
            [(pixel_x, pixel_y), skip_marker] = myGUI.current_marker
            # BACK PROJECT MARKER POSITION TO CAMERA 3D COORDINATES
            marker_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
            x, y = marker_position[0,0], marker_position[1,0]
            caption = f"Position:{x:.2f},{y:.2f}"
            myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+25)), 0.35)

            for marker in myGUI.markers:
                [(pixel_x, pixel_y), skip_marker] = marker
                # BACK PROJECT MARKER POSITION TO CAMERA 3D COORDINATES
                marker_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
                x, y, z = (position[0] for position in marker_position)
                caption = f"Position:{x:.2f},{y:.2f}"
                myGUI.drawText(myGUI.screen, caption, (int(pixel_x-25), int(pixel_y+25)), 0.35)

        cv2.imshow(WINDOW_TITLE,myGUI.screen)

        # KEYBOARD COMMANDS
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        if quit:
            break

        cv2.setMouseCallback(WINDOW_TITLE, myGUI.pointCrossMarker)
        myGUI.updateGUI(img)
    
    cv2.destroyAllWindows()