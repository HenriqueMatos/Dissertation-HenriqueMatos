from cv2 import norm
import numpy as np


p1 = np.array([0, 0])
p2 = np.array([0, 500])
p3 = np.array([250, 250])

d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)


print(d)

print(np.linalg.norm(p2-p1))


def isGoingInsideFrame(CenterPoint, InicialPoint, FinalPoint):
    center_point = np.array(CenterPoint)
    inicial_point = np.array(InicialPoint)
    final_point = np.array(FinalPoint)

    inicial_distance = np.linalg.norm(center_point-inicial_point)
    final_distance = np.linalg.norm(center_point-final_point)
    # está a afastar-se
    print(final_distance, inicial_distance)
    if inicial_distance > final_distance:
        return False
    else:
        return True


print(isGoingInsideFrame([250, 250], [200, 0], [100, 0]))
