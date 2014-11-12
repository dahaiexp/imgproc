import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image in grayscale
#gray = cv2.imread('../cpp/fruits.jpg',0)
gray = cv2.imread('./data/fruits2.jpg',0)
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)

#img=cv2.drawKeypoints(gray,kp)
plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
#cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
