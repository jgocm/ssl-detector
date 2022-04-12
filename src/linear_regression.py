from sklearn import linear_model
import pandas as pd
import numpy as np
import os
import interface
import object_localization
import cv2

def predictLinearRegression(xmin, xmax, ymin, ymax, x_weights, y_weights):
    x = [xmin, xmax, ymin, ymax, 1]@x_weights
    y = [xmin, xmax, ymin, ymax, 1]@y_weights
    return x, y

if __name__=="__main__":
    WINDOW_TITLE = "Linear Regression"

    cwd = os.getcwd()

    # LOAD IMAGE
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
    
    # SHOW LINEAR REGRESSION RESULTS ON SCREEN
    regression_weights = np.loadtxt(cwd+f"/experiments/9abr/regression_weights.txt")
    bboxes = np.loadtxt(cwd+f"/experiments/9abr/bounding_boxes.txt")

    for bounding_box in bboxes:
        [xmin, xmax, ymin, ymax] = bounding_box
        pixel_x, pixel_y = predictLinearRegression(xmin, xmax, ymin, ymax, regression_weights[0], regression_weights[1])
        marker = [(int(pixel_x), int(pixel_y)), False]
        myGUI.placeCrossMarker(marker)

    while True:
        
        myGUI.runUI(myGUI.screen)

        if myGUI.mode == 'debug':
            [(pixel_x, pixel_y), skip_marker] = myGUI.current_marker
            # BACK PROJECT MARKER POSITION TO CAMERA 3D COORDINATES
            marker_position = ssl_cam.pixelToCameraCoordinates(x=pixel_x, y=pixel_y, z_world=0)
            x, y, z = (position[0] for position in marker_position)
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