import math
from entities import Field

class ParticleVision:
    '''
    Class for simulating SSL Embedded vision module
    '''
    def __init__(
                self,
                vertical_lines_offset = 320,
                input_height = 640,
                input_width = 480
                ):
        self.vertical_lines_offset = vertical_lines_offset
        self.vertical_scan_angles = [0]

    def project_line(self, x, y, theta):
        coef = math.tan(math.radians(theta))
        intercept = y - coef*x
        return coef, intercept

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def intercept_upper_boundary(self, a, b, field):
        y = field.y_max
        x = (y-b)/a
        return x, y

    def intercept_left_boundary(self, a, b, field):
        x = field.x_min
        y = a*x + b
        return x, y

    def intercept_lower_boundary(self, a, b, field):
        y = field.y_min
        x = (y-b)/a
        return x, y

    def intercept_right_boundary(self, a, b, field):
        x = field.x_max
        y = a*x + b
        return x, y

    def intercept_field_boundaries(self, x, y, line_dir, field):
        a, b = self.project_line(x, y, line_dir)

        if 0 == line_dir:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            return x1, y1
        elif 90 == line_dir:
            x1, y1 = self.intercept_upper_boundary(a, b, field)
            return x1, y1
        elif 180 == line_dir or -180 == line_dir:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            return x1, y1
        elif -90 == line_dir:
            x1, y1 = self.intercept_lower_boundary(a, b, field)
            return x1, y1

        elif 0 < line_dir and line_dir < 90:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            x2, y2 = self.intercept_upper_boundary(a, b, field)
        elif 90 < line_dir and line_dir < 180:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            x2, y2 = self.intercept_upper_boundary(a, b, field)
        elif -180 < line_dir and line_dir < -90:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            x2, y2 = self.intercept_lower_boundary(a, b, field)
        elif -90 < line_dir and line_dir < 0:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            x2, y2 = self.intercept_lower_boundary(a, b, field)

        if self.get_distance(x, y, x1, y1) < self.get_distance(x, y, x2, y2):
            return x1, y1
        else:
            return x2, y2

    def convert_to_local(self, global_x, global_y, robot_x, robot_y, theta):
        x = global_x - robot_x
        y = global_y - robot_y
        theta = math.radians(theta)
        x, y = x*math.cos(theta) + y*math.sin(theta),\
            -x*math.sin(theta) + y*math.cos(theta)

        return x, y

    def detect_boundary_points(self, x, y, w, field):
        '''
        Computes x, y relative position of a boundary point on robot's direction
        based on the current robot's position and field dimensions

        params:
        x: robot global x position in meters
        y: robot global y position in meters
        w: robot global orientation in degrees
        field: a field object with properties x_max, y_max, x_min, y_min
        ------------------------
        returns:
        boundary_x: boundary point x position relative to robot axis
        boundary_y: boundary point y position relative to robot axis
        '''
        line_dir = w
        interception_x, interception_y = self.intercept_field_boundaries(x, y, line_dir, field)
        interception_x, interception_y = self.convert_to_local(interception_x, interception_y, x, y, w)
        return interception_x, interception_y

if __name__ == "__main__":
    
    # SETUP PARTICLE VISION
    vision = ParticleVision()

    # SET FIELD DIMENSIONS
    field = Field()
    field.redefineFieldLimits(x_max=4, y_max=3, x_min=-0.5, y_min=-3)

    boundary_points = vision.detect_boundary_points(0, 0, 37, field)
    print(boundary_points)
    