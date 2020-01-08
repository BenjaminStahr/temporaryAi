import cv2
import cvlib as cv

# diameter of the tennis ball in cm:
KNOWN_SIZE = 6.7
# distance in cm from the tennis ball in calibration image:
KNOWN_DISTANCE = 30
# use if calibration was done with camera with different pixel size
FACTOR_FROM_RESIZE = 0.16


def calibrate_camera():
    img = cv2.imread('images/calibration_30.jpg')
    bbox, label, conf = cv.detect_common_objects(img)
    width = bbox[label.index('orange')][2] - \
            bbox[label.index('orange')][0]
    width *= FACTOR_FROM_RESIZE
    focal_length = (width * KNOWN_DISTANCE) / KNOWN_SIZE
    return focal_length


def calc_distance(focal_length, width):
    distance = (KNOWN_SIZE * focal_length) / width
    return distance
