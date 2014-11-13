import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt
from math import sqrt

def height(img):
  return img.shape[0]

def width(img):
  return img.shape[1]

def is_close(x1, x2, dx):
  if x1>x2:
    return (x1-x2<dx)
  else:
    return (x2-x1<dx)

def circle_center_x(circle):
  return circle[0]

def circle_center_y(circle):
  return circle[1]

def circle_radius(circle):
  return circle[2]

def dist(x1,y1,x2,y2):
  return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def on_circles(x,y,circles):
  first=True;
  for cir in circles[0,:]:
    if(is_close(
      dist(y,x,circle_center_x(cir),circle_center_y(cir)),
      circle_radius(cir),
       1)):
      return True
    #if(first): break
  return False

# Load an color image in grayscale
img_ori = cv2.imread('./data/fruits2.jpg')
img = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(img,7)
img_dog=img-img_blur
img_lines = img_ori.copy()

# for each 5x5 block, calculate hough line
for block_y in range(0,width(img)/5-1):
  for block_x in range(0,height(img)/5-1):
    roi = img_dog[block_x*5:block_x*5+4,block_y*5:block_y*5+4]
    roi_line = img_ori[block_x*5:block_x*5+4,block_y*5:block_y*5+4]
    #edges = cv2.Canny(roi,50,150,apertureSize = 3)
    lines = cv2.HoughLines(roi,1,np.pi/180,200*7)
    #if(lines is not None and lines.any()):
    if (lines is not None):
      for rho,theta in lines[0]:
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 5*(-b))
          y1 = int(y0 + 5*(a))
          x2 = int(x0 - 5*(-b))
          y2 = int(y0 - 5*(a))
          cv2.line(roi_line,(x1,y1),(x2,y2),(0,0,255),2)
          break;

    img_lines[block_x*5:block_x*5+4,block_y*5:block_y*5+4] = roi_line
    img[block_x*5:block_x*5+4,block_y*5:block_y*5+4] = roi
    #img_blur[block_x*5:block_x*5+4,block_y*5:block_y*5+4] = edges


plt.subplot(221),plt.imshow(img_ori,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.subplot(222),plt.imshow(img_blur,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_dog,cmap = 'gray')
plt.title('DOG Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img_lines,cmap = 'gray')
plt.title('Line Image'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

