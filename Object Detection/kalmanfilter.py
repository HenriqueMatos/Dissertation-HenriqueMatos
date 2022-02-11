# https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
from time import sleep
from black import main
import cv2
import numpy as np
from sympy import true


class KalmanFilter:

    def __init__(self, num_mean=8, num_measurement=4, uncertainty=10):
        self.hasInicialValues = False
        self.oldTime = None
        # Mean
        # [cx, cy, w, h, vx, vy, vw, vh]
        self.num_mean = num_mean
        # Measurement Vector
        # [cx, cy, w, h]
        self.num_measurement = num_measurement

        self.uncertainty = uncertainty

        self.kf = cv2.KalmanFilter(self.num_mean, self.num_measurement)

        # Covariance (P)
        self.kf.errorCovPost = self.uncertainty*np.identity(self.num_mean)

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)

        self.kf.measurementNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, self.uncertainty, 0],
                                                [0, 0, 0, self.uncertainty]], np.float32)

        out = np.identity(self.num_mean, dtype=np.float32)
        for index in range(self.num_measurement):
            out[index][self.num_measurement+index] = 1

        self.kf.transitionMatrix = out

        # kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # kf.measurementMatrix = 10*np.identity(num_elements, dtype=np.float32)

    def set_transitionMatrix(self, timeDifference):
        out = np.identity(self.num_mean, dtype=np.float32)
        for index in range(self.num_measurement):
            out[index][self.num_measurement+index] = timeDifference

        self.kf.transitionMatrix = out

    def correct_kalman_filter(self, cx, cy, w, h, timeDifference):
        if self.oldTime is None:
            timeVariance = 0.0
        else:
            timeVariance = timeDifference-self.oldTime
        self.hasInicialValues = True
        self.set_transitionMatrix(timeVariance)
        measured = np.array(
            [np.float32(cx), np.float32(cy), np.float32(w), np.float32(h)])

        self.kf.correct(measured)

    def predict(self):

        # self.correct_kalman_filter(cx, cy, w, h, timeDifference)
        # self.set_transitionMatrix(timeDifference)

        # measured = np.array(
        #     [np.float32(cx), np.float32(cy), np.float32(w), np.float32(h)])

        # self.kf.correct(measured)
        predicted = self.kf.predict()

        final = []
        for item in predicted.tolist():
            final.append(round(item[0], 4))
        cx = final[0]
        cy = final[1]
        w = final[2]
        h = final[3]
        vx = final[4]
        vy = final[5]
        vw = final[6]
        vh = final[7]
        return cx, cy, w, h, vx, vy, vw, vh


# def main():
#     kf = KalmanFilter()

#     for item in range(100):
#         kf.correct_kalman_filter(item, item, 10, 10, 1.0)
#         predicted = kf.predict(item, item, 10, 10, 1.0)

#     print(predicted)
#     kf.correct_kalman_filter(500, 500, 40, 10, 1.0)
#     predicted = kf.predict(500, 500, 40, 10, 1.0)
#     print(predicted)


# if __name__ == '__main__':
#     main()
