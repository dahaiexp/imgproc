import numpy as np
import cv2
from matplotlib import pyplot as plt

def height(img):
  return img.shape[0]

def width(img):
  return img.shape[1]

def draw_borders_in_middle(img, rect_width, plt):
    print "draw_borders_in_middle:width="+str(width(img))+";height="+str(height(img))
    for x in range(0, width(img), rect_width):
        plt.plot([x, 0], [x, height(img)], 'k-')
    for y in range(0, height(img), rect_width):
        plt.plot([0, y], [width(img), y], 'k-')
    return


# Load an color image in grayscale
#gray = cv2.imread('../cpp/fruits.jpg',0)
gray = cv2.imread('./data/fruits2.jpg',0)
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# split the image into 7x7 blocks
plt.imshow(gray, cmap = 'gray', interpolation = 'nearest')
draw_borders_in_middle(gray, 7, plt)

# apply hough transform in these blocks
# draw the lines found

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)

#img=cv2.drawKeypoints(gray,kp)
#plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
#cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

