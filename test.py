import cv2

img = cv2.imread('2.jpg')
while(1):
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    if k==27:
        break
    elif k==-1:
        continue
    else:
        print(k)