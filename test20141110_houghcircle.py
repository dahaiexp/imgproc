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
img = cv2.imread('./data/fruits2.jpg',0)
img = cv2.medianBlur(img,3)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,5, param1=100,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

first=True;
for cir in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(cir[0],cir[1]),cir[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(cir[0],cir[1]),2,(0,0,255),3)
    #if(first): break

edges = cv2.Canny(img,100,50)
img_circle = cv2.imread('./data/fruits2.jpg')
for y in range(0,width(img_circle)-2):
  for x in range(0,height(img_circle)-2):    
    if (on_circles(x,y, circles)):
      #img_circle.itemset((x,y,1),255)
      img_circle[x,y]=edges[x,y]
    else:
      img_circle.itemset((x,y,0),0)
      img_circle.itemset((x,y,1),0)
      img_circle.itemset((x,y,2),0)



plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.subplot(222),plt.imshow(edges,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(cimg,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img_circle,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

