import numpy as np
import cv2

def mark_point(event,x,y,flags,params):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        print(x,' ',y)

        ix, iy = x, y
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(x) + ',' +
                    str(y), (x,y), font,
                    0.8, (0,0,0), 2)
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        

        #cv2.imshow('image',img)

    return x,y

if __name__=="__main__":

    img = cv2.imread('/home/joao/ssl-detector/23.jpg',1)
    #input = cv2.VideoCapture('/dev/video0')
    line_x = 320
    line_y = 240

    ix, iy = -1, -1

    while True:
        #_,img = input.read()
        key=cv2.waitKey(1) & 0xFF

        cv2.line(img, (line_x,0), (line_x,2*line_y), (0,0,0), 1)
        cv2.line(img, (0,line_y), (2*line_x,line_y), (0,0,0), 1)
        cv2.imshow('image',img)
        cv2.setMouseCallback('image',mark_point)

        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('ssl.jpg',img)        

    cv2.destroyAllWindows()