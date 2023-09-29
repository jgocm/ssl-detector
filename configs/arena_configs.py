import numpy as np

# ARENA DIMENSIONS
field_length = 1800
field_width = 900

# FIELD POINTS
reference_points = [
    'Arena Lower Left Corner',
    'Arena Lower Right Corner',
    'Arena Upper Left Corner',
    'Arena Upper Right Corner',
]

p1 = (0, 0, 0)
p2 = (field_length, 0, 0)
p3 = (0, field_width, 0)
p4 = (field_length, field_width, 0)

points3d = np.array([
                    p1, p2, p3, p4
                    ],dtype="float64")

np.savetxt(f'arena_points3d.txt',points3d)
