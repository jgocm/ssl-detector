import numpy as np
import enum
import cv2

class GUI():
    def __init__(
                self,
                img=None,
                display_menu=False,
                play=True,
                mode='detection'
                ):
        super(GUI,self).__init__()
        self.img = img

        self.display_menu = display_menu

        self.play = play
        self.reset = False
        self.mode = 'detection'
        self.state = 'Idle'

        self.objToMark = None
        self.objToDetect = [0,0,0]

        self.marker_points = []
        self.marker_color = None
        self.marker_position = None

        self.skip_point = False

    def markPoint(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            if self.mode=='marker':
                item=self.objToMark
                color=self.marker_color
                self.marker_position = (x,y)
                point=[(x,y),(color),item]
                self.marker_points.append(point)
            elif self.mode=='calibration' and self.isCollecting():
                item=self.objToMark
                color=self.marker_color
                self.marker_position = (x,y)
                point=[(x,y),(color),item]
                self.marker_points.append(point)

        return x,y
    
    def goalAsPoints(self, left, top, right, bottom):
        if self.mode=='detection':
            goalTop = 0.9*top+0.1*bottom
            goalBottom = 0.1*top+0.9*bottom
            goalLeft = 0.85*left+0.15*right
            goalRight = 0.15*left+0.85*right
            goal2dCoordinates = np.array([
                            (goalLeft,goalTop),
                            (goalRight,goalTop),
                            (goalLeft,goalBottom),
                            (goalRight,goalBottom)],
                            dtype="float64")
            return goal2dCoordinates

    def ballAsPoint(self, left, top, right, bottom):
        if self.mode=='detection':
            x = (left+right)/2
            y = 0.9*bottom+0.1*top
            return x, y 
    
    def robotAsPoint(self, left, top, right, bottom):
        if self.mode=='detection':
            x = (left+right)/2
            y = 0.9*bottom+0.1*top
            return x, y

    def drawDetectionMarker(self, item, color, point, pointToCamera, pointToField, flag):
        x, y = point
        x, y = int(x),int(y)
        color = color
        item = item
        point_marker = [(x,y), (color), item]
        self.marker_points.append(point_marker)
        self.drawMarkerPoint(point=point_marker,
                            pointToCamera=pointToCamera,
                            pointToField=pointToField,
                            flag=flag)

    def setState(self):
        title = None
        if self.reset:
            self.play = True
            #self.mode = 'marker'
            self.marker_points = []
            self.objToDetect = [0,0,0]
            self.objToMark = None
            self.reset = False
            
        if self.play:
            title = 'Play'
        else:
            title = 'Pause'

        if self.mode == 'detection':
            title = title + ' | Object Detection'
            self.marker_points = []
        elif self.mode == 'marker':
            title = title + ' | Point Marker'
        elif self.mode == 'calibration':
            title = title + ' | Calibration'
            self.objToDetect=[0,0,0]
        
        if self.objToDetect==[0,0,0]:
            self.marker_color = (0,0,0)
            self.objToMark = None

        if self.objToDetect[0]:
            if self.objToMark == 'goal':
                self.marker_color = (0,255,0)   # GOAL = GREEN
            title = title + ' | Goal'

        if self.objToDetect[1]:
            if self.objToMark == 'ball':
                self.marker_color = (255,0,0)   # BALL = BLUE
            title = title + ' | Ball'
            
        if self.objToDetect[2]:
            if self.objToMark == 'robot':
                self.marker_color = (0,0,255)   # ROBOT = RED
            title = title + ' | Robot'

        return title

    def showTitle(self):
        title = self.setTitle()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.img, title, (0,30), font,
                    1, self.marker_color, 2)

    def showMenu(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        leftOrg = (5,195)
        rightOrg = (180,185)
        leftFontScale = 0.5
        rightFontScale = 0.4
        color = (0,0,0)
        thickness = 2

        if self.display_menu:
            cv2.putText(self.img, 
                        'PLAY(p)', 
                        org=(5,195), 
                        fontFace=font, 
                        fontScale=0.5, 
                        color=(0,0,0),
                        thickness=2)

            cv2.putText(self.img, 
                        'QUIT(q)', 
                        org=(5,215), 
                        fontFace=font, 
                        fontScale=0.5, 
                        color=(0,0,0),
                        thickness=2)

            cv2.putText(self.img, 
                        'GOAL(g)', 
                        org=(165,185), 
                        fontFace=font, 
                        fontScale=0.4, 
                        color=(0,0,0),
                        thickness=2)

            cv2.putText(self.img, 
                        'BALL(b)', 
                        org=(165,200), 
                        fontFace=font, 
                        fontScale=0.4, 
                        color=(0,0,0),
                        thickness=2)

            cv2.putText(self.img, 
                        'ROBOT(r)', 
                        org=(165,215), 
                        fontFace=font, 
                        fontScale=0.4, 
                        color=(0,0,0),
                        thickness=2)
    
    def drawMarkerPoint(self, point, pointToCamera, pointToField, flag):
        font = cv2.FONT_HERSHEY_SIMPLEX
        [(x,y), (color), item] = point
        cv2.circle(self.img,(x,y),3,color,-1)
        cv2.putText(self.img,'Pixel:' + str(x) + ',' + str(y), 
                    (x+5,y-5), font,
                    0.4, color, 1)
        cv2.putText(self.img,'Point to Camera:' + str(int(pointToCamera[0])) + ',' + str(int(pointToCamera[1])), 
                    (x+5,y+5), font,
                    0.4, color, 1)
        if flag==True:
            cv2.putText(self.img,'Point to Field:' + str(int(pointToField[0])) + ',' + str(int(pointToField[1])), 
                        (x+5,y+15), font,
                        0.4, color, 1)
        return self.img

    def makeCalibrationUI(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.skip_point==True:
            null_point = [(-1,-1), (0,0,0), None]
            self.marker_points.append(null_point)
            self.skip_point=False

        for point in self.marker_points:
            [(x,y), (color), item] = point
            if x>0:
                cv2.circle(self.img,(x,y),3,color,-1)
                cv2.putText(self.img,'Pixel:' + str(x) + ',' + str(y), 
                            (x+5,y-5), font,
                            0.4, color, 1)

        n = len(self.marker_points)
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

        if n>9:
            cv2.putText(self.img,titles[10] + ' ('+ str(10)+'/10)', 
                    (0,20), font,
                    0.8, (0,0,0), 2)
            return False
        cv2.putText(self.img,titles[n] + ' ('+ str(n)+'/10)', 
                    (0,20), font,
                    0.8, (0,0,0), 2)
        cv2.putText(self.img,'skip point: right arrow', 
            (0,40), font,
            0.8, (0,0,0), 2)
        return True

    def isCollecting(self):
        return self.makeCalibrationUI()

    def updateImg(self, img):
        self.img = img

    def updateGUI(self, img):
        img = self.img
        self.state = self.setState()
        #self.showTitle()
        #self.showMenu()

        return img

    def commandHandler(self, key):
        key = key & 0xFF

        if key == ord('c'):
            self.reset = True

        if key == ord('p'):
            self.play = 1-self.play

        if key == ord('m'):
            self.objToDetect = [0,0,0]
            self.marker_points = []
            if self.mode == 'detection':
                self.mode = 'marker'
            elif self.mode == 'marker':
                self.mode = 'calibration'
            elif self.mode == 'calibration':
                self.mode = 'detection'

        if key == ord('g'):
            self.objToDetect[0] = 1 - self.objToDetect[0]
            if self.mode=='marker':
                self.objToMark = 'goal'
                self.objToDetect[1] = 0
                self.objToDetect[2] = 0

        if key == ord('b'):
            self.objToDetect[1] = 1 - self.objToDetect[1]
            if self.mode=='marker':
                self.objToMark = 'ball'
                self.objToDetect[0] = 0
                self.objToDetect[2] = 0

        if key == ord('r'):
            self.objToDetect[2] = 1 - self.objToDetect[2]
            if self.mode=='marker':
                self.objToMark = 'robot'
                self.objToDetect[0] = 0
                self.objToDetect[1] = 0

        if key == ord('a'):
            self.objToDetect = [1,1,1]
        
        if key == ord('n'):
            self.objToDetect = [0,0,0]

        if key == 83:   # right arrow key
            self.skip_point = True
        
        return key


if __name__=="__main__":
    # OpenCV configs
    font = cv2.FONT_HERSHEY_SIMPLEX

    img = cv2.imread('/home/joao/ssl-dataset/1_resized/00285.jpg')
    play = True
    detection = False

    myGUI = GUI(img=img, play=play, display_menu=False, detection=detection)
    #img = cv2.imread('/home/joao/ssl-dataset/1_resized/00286.jpg')

    window_title = 'image'

    while True:
        cv2.imshow(window_title,img)

        # KEYBOARD COMMANDS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            myGUI.commandHandler(key=cv2.waitKey(1))
        
        myGUI.updateGUI(img)

        cv2.setMouseCallback(window_title,myGUI.markPoint)
        
    cv2.destroyAllWindows()