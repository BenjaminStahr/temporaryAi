import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


img = cv2.imread('images/calibration_30.jpg')

scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim)
bbox, label, conf = cv.detect_common_objects(img)
print(conf)
output_image = draw_bbox(img, bbox, label, conf)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
