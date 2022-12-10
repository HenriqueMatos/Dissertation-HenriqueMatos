
import math
maz_size = [2000, 2000]


def scale(w, h, x, y, maximum=True):
    nw = y * w / h
    nh = x * h / w
    if maximum ^ (nw >= x):
        return nw or 1, y
    return x, nh or 1

def rotate(xy, theta):
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    return (
        xy[0] * cos_theta - xy[1] * sin_theta,
        xy[0] * sin_theta + xy[1] * cos_theta
    )

def translate(xy, offset):
    return xy[0] + offset[0], xy[1] + offset[1]

def get_corresponding_coord(local_coord, cam_coordinates, scale, offset, degree=0):
    new_coord = [(local_coord[0]*scale[0])/cam_coordinates[0],
                 (local_coord[1]*scale[1])/cam_coordinates[1]]

    new_rotate_coord = rotate(new_coord, math.radians(degree))

    final_coord = translate(new_rotate_coord, offset)

    return final_coord



# print(scale(1280, 720, 500, 500))

# print(rotate(scale(1280, 720, 500, 500), math.radians(90)))

# print(translate(rotate(scale(1280, 720, 500, 500), math.radians(360)), (100, 100)))





print(get_corresponding_coord([750, 460], [1280, 720], scale(1280, 720, 500, 500), [100, 100]))

print(get_corresponding_coord([1200,720], [1280, 720], scale(1280, 720, 500, 500), [0, 0],0))
