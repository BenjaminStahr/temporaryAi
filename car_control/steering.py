import numpy as np
import cv2
from SunFounder_PiCar.picar import front_wheels
from SunFounder_PiCar.picar import back_wheels
import SunFounder_PiCar.picar as picar

picar.setup()

fw = front_wheels.Front_Wheels(db='config')
bw = back_wheels.Back_Wheels(db='config')
fw.turning_max = 45

ball_detected_speed = 50


def steering(position, distance, framesize):
    rel_position = (framesize / 2) - position
    angle = np.degrees(np.arctan(np.abs(rel_position) / distance))
    if rel_position < 0:
        drive_angle = 90 + angle
    else:
        drive_angle = 90 - angle

    fw.turn(drive_angle)
    bw.forward()
    bw.speed(ball_detected_speed)


def steer_to_bbox(bbox, frame):
    hpos = (np.size(frame, 1) / 2) - (bbox[0] + (bbox[2] - bbox[0]))
    vpos = (np.size(frame, 2) / 2) - (bbox[1] + (bbox[1] - bbox[3]))
    angle = np.degrees(np.arctan(np.abs(hpos) / vpos))
    if hpos < 0:
        drive_angle = 90 + angle
    else:
        drive_angle = 90 - angle
    fw.turn(drive_angle)
