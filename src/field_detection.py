import numpy as np
import cv2
import os

class FieldDetection():
    def __init__(
            self,
            vertical_lines_offset = 240,
            min_line_length = 1,
            max_line_length = 20,
            min_wall_length = 10
            ):
        # DEFINE COLORS:
        self.BLACK = [0, 0, 0]
        self.BLUE = [255, 0, 0]
        self.GREEN = [0, 255, 0]
        self.RED = [0, 0, 255]
        self.WHITE = [255, 255, 255]

        # min/max amount of pixels for detection
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length
        self.min_wall_length = min_wall_length

        # line scans offset
        self.vertical_lines_offset = int(vertical_lines_offset)

        self.mask_points = []
    
    def isBlack(self, src):
        blue, green, red = src
        if green < 50 and red < 50 and blue < 50:
            return True
        else:
            return False
    
    def isGreen(self, src):
        blue, green, red = src
        if green > 90 and red < 110:
            return True
        else:
            return False     

    def isWhite(self, src):
        blue, green, red = src
        if blue > 130 and green > 130 and red > 130:
            return True
        else:
            return False

    def segmentPixel(self, src):
        if self.isWhite(src):
            color = self.WHITE
            return color        
        elif self.isBlack(src):
            color = self.BLACK
            return color
        elif self.isGreen(src):
            color = self.GREEN
            return color
        else:
            return src
    
    def segmentField(self, src):
        """
        Make description here
        """
        # make copy from source image for segmentation
        # segmented_img = src.copy()
        segmented_img = src

        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        for line_x in range(self.vertical_lines_offset+5, width, self.vertical_lines_offset):
             # segment vertical lines
            for pixel_y in range(0, height):
                pixel = src[pixel_y, line_x]
                color = self.segmentPixel(pixel)
                segmented_img[pixel_y, line_x] = color

        return segmented_img        
            
    def fieldWallDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # wall detection points
        boundary_points = []

        for line_x in range(self.vertical_lines_offset+5, width, self.vertical_lines_offset):
            wall_points = []
            for pixel_y in range(height-1, 0, -1):
                pixel = src[pixel_y, line_x]
                if len(wall_points)>self.min_wall_length:
                    boundary_points.append(wall_points[0])
                    break
                elif self.isBlack(pixel):
                    wall_points.append([pixel_y, line_x])
                else:
                    wall_points = []

        return boundary_points

    def fieldLineDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # field lines detection points
        field_line_points = []

        for line_x in range(self.vertical_lines_offset+5, width, self.vertical_lines_offset):
            field_line = False
            line_points = []
            for pixel_y in range(height-1, 0, -1):
                pixel = src[pixel_y, line_x]
                if not self.isBlack(pixel):
                    # check white->green border
                    if not self.isGreen(pixel) and field_line == False:
                        field_line = True
                    # check white->green border
                    elif self.isGreen(pixel) and field_line == True:
                        field_line = False
                        # check white line length (width)
                        if len(line_points)>self.min_line_length and len(line_points)<self.max_line_length:
                            line_y = [point[0] for point in line_points]
                            point = int(np.mean(line_y)), line_x
                            field_line_points.append(point)
                        line_points = []

                    if field_line == True:
                        line_points.append([pixel_y, line_x])

        return field_line_points      

    def detectFieldLinesAndBoundary(self, src):
        """
        Make description here
        """
        segmented_img = self.segmentField(src)
        boundary_points = self.fieldWallDetection(segmented_img)
        field_line_points = self.fieldLineDetection(segmented_img)
        return boundary_points, field_line_points     

    def detectFieldLinesAndBoundaryMerged(self, src):
        """
        Make description here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # wall detection points
        boundary_points = []

        # field lines detection points
        field_line_points = []

        for line_x in range(self.vertical_lines_offset+5, width, self.vertical_lines_offset):
            wall_points = []
            for pixel_y in range(height-1, 0, -1):
                pixel = src[pixel_y, line_x]
                if len(wall_points)>self.min_wall_length:
                    boundary_points.append(wall_points[0])
                    break
                elif self.isBlack(pixel):
                    wall_points.append([pixel_y, line_x])
                else:
                    wall_points = []
        
        return boundary_points, field_line_points       


if __name__ == "__main__":

    cwd = os.getcwd()

    FRAME_NR = 5
    STAGE = 2
    IMG_PATH = cwd + f'/data/stage{STAGE}/frame{FRAME_NR}.jpg'
    WINDOW_NAME = "BOUNDARY DETECTION"
    VERTICAL_LINES_NR = 1

    img = cv2.imread(IMG_PATH)
    height, width = img.shape[0], img.shape[1]
    alpha = 0.01

    # FIELD DETECTION TESTS
    field_detector = FieldDetection(
                    vertical_lines_offset=1,
                    min_line_length=1,
                    max_line_length=20,
                    min_wall_length=10)

    while True:
        boundary_points, line_points = field_detector.detectFieldLinesAndBoundary(img)

        for point in boundary_points:
            pixel_y, pixel_x = point
            img[pixel_y, pixel_x] = field_detector.RED

        for point in line_points:
            pixel_y, pixel_x = point
            img[pixel_y, pixel_x] = field_detector.RED
            
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break

    # RELEASE WINDOW AND DESTROY
    cv2.destroyAllWindows()