import numpy as np
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from recognition.distance import calc_distance, calibrate_camera
from car_control.steering import steering

# cap = cv2.VideoCapture('output.avi')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 3)
focal_length = calibrate_camera()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # scale_percent = 50  # percent of original size
    # width = int(frame.shape[1] * scale_percent / 100)
    # height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # frame = cv2.resize(frame, dim)

    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)

    width = 0
    position = 0
    left_pos = 0
    right_pos = 0
    side = None

    if 'sports ball' in label:
        right_pos = bbox[label.index('sports ball')][2]
        left_pos = bbox[label.index('sports ball')][0]
    if 'apple' in label:
        right_pos = bbox[label.index('apple')][2]
        left_pos = bbox[label.index('apple')][0]
    if 'orange' in label:
        right_pos = bbox[label.index('orange')][2]
        left_pos = bbox[label.index('orange')][0]
    if 'banana' in label:
        right_pos = bbox[label.index('banana')][2]
        left_pos = bbox[label.index('banana')][0]
    width = right_pos - left_pos
    position = left_pos + (width / 2)
    if width != 0:
        distance = calc_distance(focal_length, width)
        print(distance)
        steering(position, distance, np.size(frame, 1))
    else:
        # TODO Find Ball to detect
        pass

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
