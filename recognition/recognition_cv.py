import cv2
import numpy as np
import matplotlib.pyplot as plt

# lower_bound1 = np.array([0, 50, 150])
# upper_bound1 = np.array([25, 255, 255])
#
# lower_bound2 = np.array([335, 50, 150])
# upper_bound2 = np.array([360, 255, 255])

lower_bound = np.array([200, 0, 0])
upper_bound = np.array([255, 100, 100])

img = cv2.imread("images/3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (400, 300))
# img1 = cv2.inRange(img, lower_bound1, upper_bound1)
# img2 = cv2.inRange(img, lower_bound2, upper_bound2)

img = cv2.inRange(img, lower_bound, upper_bound)

img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

# img = cv2.addWeighted(img1, 1, img2, 1, 0)

plt.imshow(img, cmap="gray")
plt.show()


