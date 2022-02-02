# A Python3 program to find if 2 given line segments intersect or not


# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'


def onSegment(p, q, r):
    (p_x, p_y) = p
    (q_x, q_y) = q
    (r_x, r_y) = r
    if ((q_x <= max(p_x, r_x)) and (q_x >= min(p_x, r_x)) and
            (q_y <= max(p_y, r_y)) and (q_y >= min(p_y, r_y))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www_geeksforgeeks_org/orientation-3-ordered-points/amp/
    # for details of below formula_
    (p_x, p_y) = p
    (q_x, q_y) = q
    (r_x, r_y) = r
    val = (float(q_y - p_y) * (r_x - q_x)) - (float(q_x - p_x) * (r_y - q_y))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Collinear orientation
        return 0

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect_


def doIntersect(p1, q1, p2, q2):

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    # print(o1)
    # print(o2)
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


# # Driver program to test above functions:
# p1 = (1, 1)
# q1 = (10, 1)
# p2 = (1, 2)
# q2 = (10, 2)

# if doIntersect(p1, q1, p2, q2):
#     print("Yes")
# else:
#     print("No")

# p1 = (10, 0)
# q1 = (0, 10)
# p2 = (0, 0)
# q2 = (10, 10)

# if doIntersect(p1, q1, p2, q2):
#     print("Yes")
# else:
#     print("No")

# p1 = (2, 2)
# q1 = (5, 5)
# p2 = (1, 4)
# q2 = (2, 3)


# if doIntersect(p1, q1, p2, q2):
#     print("Yes")
# else:
#     print("No")
