import numpy as np
import cv2
import os
from object_localization import Camera

def is_black(blue, green, red):
    if green > 120 and red < 110:
        return False
    elif green < 50 and red < 50 and blue < 50:
        return True
    else:
        return False

if __name__ == "__main__":

    cwd = os.getcwd()

    FRAME_NR = 132
    IMG_PATH = cwd + f'/data/stage1/frame{FRAME_NR}.jpg'
    WINDOW_NAME = "BOUNDARY DETECTION"
    VERTICAL_LINES_NR = 1

    # DEFINE COLORS:
    BLACK = [0, 0, 0]
    BLUE = [255, 0, 0]
    GREEN = [0, 255, 0]
    RED = [0, 0, 255]
    WHITE = [255, 255, 255]

    img = cv2.imread(IMG_PATH)
    height, width = img.shape[0], img.shape[1]
    vertical_lines_offset = int(width/(1+VERTICAL_LINES_NR))

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_DISTORTION_PARAMETERS = cwd+"/configs/dist.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")
    ssl_cam = Camera(
                camera_matrix=camera_matrix,
                camera_initial_position=calibration_position
                )
    points2d = np.loadtxt(PATH_TO_2D_POINTS, dtype="float64")
    points3d = np.loadtxt(PATH_TO_3D_POINTS, dtype="float64")
    ssl_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)

    while True:

        segmented_img = img.copy()
        boundary_points = []
    
        for line_x in range(0, width, vertical_lines_offset):
            # segment vertical lines
            wall_points = []
            for pixel_y in range(height-1, 0, -1):
                blue, red, green = img[pixel_y, line_x]

                if len(wall_points)>10:
                    boundary_points.append(wall_points[0])
                    break
                elif is_black(blue, red, green):
                    wall_points.append([pixel_y, line_x])
                else:
                    wall_points = []
        
        for pixel in boundary_points:
            pixel_y, pixel_x = pixel
            if True: 
                segmented_img[pixel_y, pixel_x] = RED
                boundary_position = ssl_cam.pixelToCameraCoordinates(
                                            x = pixel_x,
                                            y = pixel_y,
                                            z_world = 0)
                x, y, z = (position[0] for position in boundary_position)
                caption = f"Position:{x:.2f},{y:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(segmented_img, caption, (int(pixel_x-50), int(pixel_y-15)), font, 0.5, (0,0,0), 2)
                cv2.putText(segmented_img, caption, (int(pixel_x-50), int(pixel_y-15)), font, 0.5, (255,255,255), 1)

        cv2.imshow(WINDOW_NAME, segmented_img)

        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break

    # RELEASE WINDOW AND DESTROY
    cv2.destroyAllWindows()