class Ball():
    def __init__(
        self,
        x = 0,
        y = 0,
        diameter = 42.7,
        radius = diameter/2
    )
    super(Ball, self).__init__()
    self.x = x
    self.y = y
    self.radius = diameter/2

    def updatePosition(self, x, y):
        self.x, self.y = x, y
    
    def getDistance(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def getDirection(self):
        return atan2(self.y, self.x)

class Goal():
    def __init__(
        self,
        width = 1000,
        depth = 180,
        height = 155
    )
    super(Goal, self).__init__()
    self.width = width
    self.depth = depth
    self.height = height

class Field():
    def __init__(
        self,
        field_width = 3760,
        field_length = 5640,
        penalty_area_width = 2000,
        penalty_area_depth = 1000,
        center_radius = 1000,
        boundary_width = 180,
        line_thickness = 20
    )
    super(Field, self).__init__()
    self.width = field_width
    self.length = field_length
    self.penalty_area_width = penalty_area_width
    self.penalty_area_depth = penalty_area_depth
    self.boundary_width = boundary_width
    self.line_thickness = line_thickness
    self.center_radius = center_radius
    self.goal = Goal()
    
    def getGoalCoordinates(self):
        p1 = -self.goal.width/2, self.length/2
        p2 = self.goal.width/2, self.length/2
        return p1, p2
    
    


