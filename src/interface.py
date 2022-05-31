import numpy as np
import cv2

class GUI():
    def __init__(
                self,
                screen = np.zeros((640,480)),
                play = True,
                mode = "calibration",
                center_lines = True
                ):
        super(GUI, self).__init__()

        self.screen = screen
        self.play = play
        self.mode = mode
        self.save = False

        self.markers = []
        self.current_marker = []
        self.create_marker = True
        self.is_calibrating = False
        self.arrow_key = None

        self.center_lines = center_lines

    def clearUI(self):
        self.markers = []
        self.current_marker = []
        self.create_marker = True
        self.is_calibrating = False
        self.arrow_key = None

    def commandHandler(self, key):
        key = key & 0xFF
        quit = False

        if key == ord('q'):
            quit = True
        if key == ord('m'):
            self.clearUI()
            if self.mode == 'calibration':
                self.mode = 'debug'
            elif self.mode == 'debug':
                self.mode = 'calibration'
        if key == ord('p'):
            self.play = 1-self.play
        if key == ord('s'):
            if self.mode == 'calibration' and (not self.is_calibrating):
                self.save = True 
        if key == ord('c'):
            if self.mode == 'debug':
                self.clearMarkers()
        
        # CALIBRATION CONTROLLER
        if key == 81 or key == 37 or key == ord('a'):   # left arrow key
            self.arrow_key = "left"
        if key == 82 or key == 38 or key == ord('w'):   # up arrow key
            self.arrow_key = "up"
        if key == 83 or key == 39 or key == ord('d'):   # right arrow key
            self.arrow_key = "right"
        if key == 84 or key == 40 or key == ord('s'):   # down arrow key
            self.arrow_key = "down"

        if key == 32:   # space key
            self.placeCrossMarker(self.current_marker)
        if key == 13:   # enter key
            self.skipCrossMarker()
        if key == 8:    # backspace key
            self.removeCrossMarker()

        return quit
    
    def drawText(self, img, caption, box, size=0.6):
        b = np.array(box).astype(int)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img, caption, (b[0], b[1]), font, size, (0, 0, 0), 2
        )
        cv2.putText(
            img, caption, (b[0], b[1]), font, size, (255, 255, 255), 1
        )

    def createCrossMarker(self, marker_x=240, marker_y=320):
        skip_marker = False
        marker = [(marker_x, marker_y),skip_marker]

        return marker

    def placeCrossMarker(self, marker):
        self.markers.append(marker)
        self.create_marker=True
    
    def skipCrossMarker(self):
        null_point = [(-1,-1), True]
        self.markers.append(null_point)
        self.create_marker=True
        
    def removeCrossMarker(self):
        last_marker = self.markers[-1:][0]
        [(marker_x, marker_y),skip_marker] = last_marker
        if skip_marker:
            self.create_marker=True
        else:
            self.current_marker = last_marker
        self.markers = self.markers[:-1]
        
    def drawCrossMarker(self, img, center_x, center_y, color=(255,0,0), display_text=True, text_size=0.4):
        marker_height = int(self.screen.shape[0]*0.015)
        marker_width = int(self.screen.shape[0]*0.015)
        caption = f"pixel:{center_x},{center_y}"
        cv2.line(img, 
                (center_x-marker_width,center_y), 
                (center_x+marker_width,center_y), 
                color,
                1)
        cv2.line(img, 
                (center_x,center_y-marker_height), 
                (center_x,center_y+marker_height), 
                color,
                1)
        if display_text:
            self.drawText(img=img,
                        caption=caption,
                        box=(center_x-3*marker_width,center_y+2*marker_height),
                        size=text_size
                        )
        
    def moveCrossMarker(self, marker):
        [(center_x, center_y),skip_marker] = marker

        if self.arrow_key == "left":
            center_x = center_x-1
        if self.arrow_key == "right":
            center_x = center_x+1
        if self.arrow_key == "up":
            center_y = center_y-1
        if self.arrow_key == "down":
            center_y = center_y+1
        
        self.arrow_key = None
        marker = [(center_x, center_y),skip_marker]

        return marker
    
    def pointCrossMarker(self, event, x, y, flags, params):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            self.current_marker = [(x, y), False]

    def clearMarkers(self):
        self.markers = []
        self.create_marker = True

    def drawMarkers(self, img):
        [(x, y),skip_marker] = self.current_marker
        self.drawCrossMarker(img, x, y, color = (0,0,255))
        for marker in self.markers:
            [(x,y), skip_marker] = marker
            if not skip_marker:
                self.drawCrossMarker(img, x, y)
    
    def makeCalibrationUI(self, img):
        titles = [
            'Field Left Corner',
            'Keeper Area Upper Left Corner',
            'Goal Bottom Left Corner',
            'Goal Bottom Center',
            'Goal Bottom Right Corner',
            'Keeper Area Upper Right Corner',
            'Field Right Corner',
            'Keeper Area Lower Left Corner',
            'Keeper Area Lower Right Corner',
            'Field Center',
            'All Points Collected'
        ]

        n = len(self.markers)
        if n>9:
            caption = 'All Points Collected ('+ str(10)+'/10)'
            done = True
        else:
            caption = titles[n] + ' ('+ str(n)+'/10)'
            if self.create_marker:
                self.current_marker = self.createCrossMarker()
                self.create_marker = False
            else:
                self.current_marker = self.moveCrossMarker(self.current_marker)
            done = False

        self.drawText(img,
            caption=caption,
            box=(0,20),
            size=0.6)
        self.drawMarkers(img)
        return done, img
    
    def makeFieldMarkerUI(self, img):
        if self.create_marker:
            self.current_marker = self.createCrossMarker()
            self.create_marker = False
        else:
            self.current_marker = self.moveCrossMarker(self.current_marker)

        done = False
        caption="Debug mode: place markers on screen"
        self.drawText(img,
            caption=caption,
            box=(0,20),
            size=0.6)
        self.drawMarkers(img)
        return done, img
    
    def runUI(self, img):
        line_x = int(img.shape[1]/2)
        line_y = int(img.shape[0]/2)
        if self.mode=="calibration":
            done, img = self.makeCalibrationUI(img)
            self.is_calibrating = 1-done
            cv2.line(img, (line_x,0), (line_x,2*line_y), (0,0,0), 1)
            cv2.line(img, (0,line_y), (2*line_x,line_y), (0,0,0), 1)
        if self.mode=="debug":
            done, img = self.makeFieldMarkerUI(img)
            
        return done
    
    def updateGUI(self, img):
        self.screen = img.copy()

if __name__=="__main__":
    WINDOW_TITLE = "test"
    cwd = os.getcwd()

    img = cv2.imread(cwd+"/configs/calibration_imag.jpg')
    img = cv2.resize(img, (640, 480))

    myGUI = GUI(
        screen=img.copy(),
    )

    while True:
        
        myGUI.runUI(myGUI.screen)
        
        cv2.imshow(WINDOW_TITLE,myGUI.screen)

        # KEYBOARD COMMANDS
        key = cv2.waitKey(10) & 0xFF
        quit = myGUI.commandHandler(key=key)
        if quit:
            break

        cv2.setMouseCallback(WINDOW_TITLE, myGUI.pointCrossMarker)
        myGUI.updateGUI(img)
    
    cv2.destroyAllWindows()