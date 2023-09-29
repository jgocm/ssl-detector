import cv2
import os

if __name__ == "__main__":
    WINDOW_TITLE = "IMG CAPTURE"

    cwd = os.getcwd()

    # USB CAMERA SETUP
    cap = cv2.VideoCapture("/dev/video2")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        if cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               print("Check video capture path")
               break
        
        # DISPLAY WINDOW
        key = cv2.waitKey(10) & 0xFF
        cv2.imshow(WINDOW_TITLE, frame)
        
        if key == ord('s'):
            cv2.imwrite(cwd+"/test_calibration.jpg", frame)
        elif key == ord('q'):
            break