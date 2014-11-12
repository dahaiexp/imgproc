import numpy as np
import cv2
from matplotlib import pyplot as plt

def height(img):
  return img.shape[0]

def width(img):
  return img.shape[1]


# Load an color image in grayscale
#gray = cv2.imread('../cpp/fruits.jpg',0)
gray = cv2.imread('./data/fruits2.jpg',0)
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray,100,200)

plt.subplot(121),plt.imshow(gray,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

