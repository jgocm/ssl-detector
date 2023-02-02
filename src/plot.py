import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

from jetson_vision import JetsonVision
from interface import GUI

def convert_to_local(global_x, global_y, robot_x, robot_y, robot_w):
    diff_x = global_x-robot_x
    diff_y = global_y-robot_y
    local_x = diff_x*np.cos(robot_w) - diff_y*np.sin(robot_w)
    local_y = diff_x*np.sin(robot_w) + diff_y*np.cos(robot_w)
    return local_x, local_y

def log_to_relative_points(df):
    ball_x = df['BALL X'].to_numpy()
    ball_y = df['BALL Y'].to_numpy()

    robot_x = df['ROBOT X'].to_numpy()
    robot_y = df['ROBOT Y'].to_numpy()
    robot_w = df['ROBOT THETA'].to_numpy()

    points_x = []
    points_y = []
    for i in range(0, len(df)):
        point_x, point_y = convert_to_local(ball_x[i], ball_y[i], robot_x[i], robot_y[i], robot_w[i])
        points_x.append(point_x)
        points_y.append(point_y)
    
    return points_x, points_y

def plot_comparison(df_raw, df_clean):
    local_x, local_y = log_to_relative_points(df_raw)
    clean_local_x, clean_local_y = log_to_relative_points(df_clean)
    plt.scatter(local_x, local_y)
    plt.scatter(clean_local_x, clean_local_y)
    plt.show()

def draw_text(img, caption, box, size=0.6):
    b = np.array(box).astype(int)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img, caption, (b[0], b[1]), font, size, (0, 0, 0), 2
    )
    cv2.putText(
        img, caption, (b[0], b[1]), font, size, (255, 255, 255), 1
    )

def draw_cross_marker(img, center_x, center_y, color=(255,0,0), display_text=True, text_size=0.6):
    marker_height = int(img.shape[0]*0.015)
    marker_width = int(img.shape[0]*0.015)
    caption = f"pixel:{center_x},{center_y}"
    cv2.line(img, 
            (center_x-marker_width,center_y), 
            (center_x+marker_width,center_y), 
            color,
            2)
    cv2.line(img, 
            (center_x,center_y-marker_height), 
            (center_x,center_y+marker_height), 
            color,
            2)
    if display_text:
        draw_text(img=img,
                    caption=caption,
                    box=(center_x-3*marker_width,center_y+2*marker_height),
                    size=text_size
                    )

if __name__ == "__main__":
    cwd = os.getcwd()

    # READ POSITION LOG FILES
    df_raw = pd.read_csv('/home/rc-blackout/ssl-detector/data/object_localization/log.csv', sep=',')
    df_clean = pd.read_csv('/home/rc-blackout/ssl-detector/data/object_localization/log_clean.csv', sep=',')
    ball_x, ball_y = log_to_relative_points(df_raw)
    # plot_comparison(df_raw, df_clean)
    
    # READ BBOX LOG FILES 
    xmin = df_raw['X_MIN'].to_list()
    xmax = df_raw['X_MAX'].to_list()
    ymin = df_raw['Y_MIN'].to_list()
    ymax = df_raw['Y_MAX'].to_list()

    # INIT JETSON EMBEDDED VISION
    vision = JetsonVision()

    # READ FRAMES LIST
    frames = df_raw['FRAME_NR'].to_list()
    valid = df_raw['VALID'].to_list()
    frame_nr = 1
    backward = False

    while frame_nr<frames[-1]:
        # ONLY SHOWS VALID FRAMES
        while not valid[frame_nr-1]:
            if backward: frame_nr -= 1
            else: frame_nr += 1

        # READ IMAGE FILES
        dir = cwd + f"/data/object_localization/{frame_nr}.jpg"

        # MAKE OPENCV WINDOW
        WINDOW_NAME = "BALL LOCALIZATION"
        img = cv2.imread(dir)
        
        # COMPUTE BALL POSITION
        jetson_vision_relative_ball = vision.trackBall(
                        score = 1,
                        xmin=xmin[frame_nr-1],
                        xmax=xmax[frame_nr-1],
                        ymin=ymin[frame_nr-1],
                        ymax=ymax[frame_nr-1])
        ssl_vision_relative_ball = [ball_x[frame_nr-1]/1000, ball_y[frame_nr-1]/1000]

        jetson_pixel_x, jetson_pixel_y = vision.jetson_cam.robotToPixelCoordinates(
                                                x=jetson_vision_relative_ball.x, 
                                                y=jetson_vision_relative_ball.y, 
                                                camera_offset=90)

        pixel_x, pixel_y = vision.jetson_cam.robotToPixelCoordinates(
                                                x=ssl_vision_relative_ball[0], 
                                                y=ssl_vision_relative_ball[1], 
                                                camera_offset=90)
        
        # DRAW ON SCREEN FOR DEBUG                                      
        draw_cross_marker(img, int(jetson_pixel_x), int(jetson_pixel_y), color=(0,255,0))
        draw_cross_marker(img, int(pixel_x), int(pixel_y))
        draw_text(img, f'SSL Vision:{ssl_vision_relative_ball[0]:.3f}, {ssl_vision_relative_ball[1]:.3f}', (10, 55), 1)
        draw_text(img, f'Jetson Vision:{jetson_vision_relative_ball.x:.3f}, {jetson_vision_relative_ball.y:.3f}', (10, 80), 1)
        cv2.rectangle(img,
                    [xmin[frame_nr-1], ymin[frame_nr-1]], 
                    [xmax[frame_nr-1], ymax[frame_nr-1]],
                    color = vision.field_detector.BLUE)
        draw_text(img, f'frame nr: {frame_nr}', (10, 30), 0.8)
        cv2.imshow(WINDOW_NAME, img)

        # KEYBOARD COMMANDS FOR DEBUG
        key = cv2.waitKey(-1) & 0xFF
        # QUIT
        if key == ord('q'):
            break
        # RETURN 1 FRAME
        elif key == ord('a') and frame_nr>1:
            frame_nr=frame_nr-1
            backward = True
        # GO TO NEXT FRAME
        else:
            frame_nr=frame_nr+1
            backward = False







