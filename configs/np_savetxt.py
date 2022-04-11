import numpy as np

# PS EYE CAMERA INTRINSIC PARAMETERS
mtx = np.array([
            (522.572,0,331.090),
            (0,524.896,244.689),
            (0,0,1)
            ])

# FIELD DIMENSIONS
field_length = 5640
field_width = 3760
goal_width = 1000
goal_depth = 180
goal_height = 155
boundary_width = 180
line_thickness = 20
penalty_area_width = 2000
penalty_area_depth = 1000

# FIELD POINTS
'''
reference_points = [
    'Field Left Corner',
    'Penalty Area Upper Left Corner',
    'Goal Bottom Left Corner',
    'Goal Bottom Center',
    'Goal Bottom Right Corner',
    'Penalty Area Upper Right Corner',
    'Field Right Corner',
    'Penalty Area Lower Left Corner',
    'Penalty Area Lower Right Corner',
    'Field Center'
]
'''
p1 = (-field_width/2, field_length/2, 0)
p2 = (-penalty_area_width/2, field_length/2, 0)
p3 = (-goal_width/2, field_length/2, 0)
p4 = (-0, field_length/2, 0)
p5 = (goal_width/2, field_length/2, 0)
p6 = (penalty_area_width/2, field_length/2, 0)
p7 = (field_width/2, field_length/2, 0)
p8 = (-penalty_area_width/2, field_length/2-penalty_area_depth, 0)
p9 = (penalty_area_width/2, field_length/2-penalty_area_depth, 0)
p10 = (0, 0, 0)

points3d = np.array([
                    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
                    ],dtype="float64")

np.savetxt(f'np_array.txt',points3d)
