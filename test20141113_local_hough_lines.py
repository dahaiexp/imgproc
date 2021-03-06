import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt
from math import sqrt
import time
from datetime import date
from datetime import datetime

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

#######################################################
#img = cv2.imread('building.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
#lines = cv2.HoughLines(edges,1,np.pi/180,50)
#if (lines is not None):
#  count = 0
#  for rho,theta in lines[0]:
#      a = np.cos(theta)
#      b = np.sin(theta)
#      x0 = a*rho
#      y0 = b*rho
#      x1 = int(x0 + 1000*(-b))
#      y1 = int(y0 + 1000*(a))
#      x2 = int(x0 - 1000*(-b))
#      y2 = int(y0 - 1000*(a))
#
#      cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#      count += 1
#      if (count>10): break
#
#plt.subplot(111),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
#plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#######################################################

#########################################################
#img = cv2.imread('building.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray_blur3 = cv2.medianBlur(gray,3)
#gray_blur7 = cv2.medianBlur(gray,7)
#gray_dog = cv2.subtract(gray, gray_blur7)
#gray_dog_out = gray_dog.copy()
#gray_dog_out = cv2.cvtColor(gray_dog_out,cv2.COLOR_GRAY2BGR)
##edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
##lines = cv2.HoughLines(gray_dog,10,np.pi/18,300)
#lines = cv2.HoughLines(gray_dog,1,np.pi/180,10)
#if (lines is not None):
#  print lines[0]
#  for i in range(0,10):
#    if (lines[0][i] is not None):
#          rho,theta = lines[0][i]
#          a = np.cos(theta)
#          b = np.sin(theta)
#          x0 = a*rho
#          y0 = b*rho
#          x1 = int(x0 + 1000*(-b))
#          y1 = int(y0 + 1000*(a))
#          x2 = int(x0 - 1000*(-b))
#          y2 = int(y0 - 1000*(a))
#          cv2.line(gray_dog_out,(x1,y1),(x2,y2),(0,0,255),2)
#
#plt.subplot(221),plt.imshow(gray,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(222),plt.imshow(gray_dog,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(223),plt.imshow(gray_dog,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(224),plt.imshow(gray_dog_out,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
#plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#########################################################

# Load an color image in grayscale
img_ori = cv2.imread('building.jpg')
img = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(img,7)
#img_dog=img-img_blur
img_dog = cv2.subtract(img, img_blur)
img_lines = img_ori.copy()

edges = cv2.Canny(img,50,150,apertureSize = 3)
#lines_all = cv2.HoughLines(edges,1,np.pi/180,200)
img_lines_all = img_ori.copy()
#if (lines_all is not None):
#  for rho,theta in lines_all[0]:
#      a = np.cos(theta)
#      b = np.sin(theta)
#      x0 = a*rho
#      y0 = b*rho
#      x1 = int(x0 + 5*(-b))
#      y1 = int(y0 + 5*(a))
#      x2 = int(x0 - 5*(-b))
#      y2 = int(y0 - 5*(a))
#      cv2.line(img_lines_all,(x1,y1),(x2,y2),(0,0,255),2)
#      break

# for each 5x5 block, calculate hough line
for block_y in range(0,width(img)/9):
  for block_x in range(0,height(img)/9):
    roi = img_dog[block_x*9:block_x*9+8,block_y*9:block_y*9+8]
    roi_line = img_ori[block_x*9:block_x*9+8,block_y*9:block_y*9+8]
    #edges = cv2.Canny(roi,50,150,apertureSize = 3)
    lines = cv2.HoughLines(roi,1,np.pi/180,10)
    #if(lines is not None and lines.any()):
    if (lines is not None):
      #for rho,theta in lines[0]:
      #    a = np.cos(theta)
      #    b = np.sin(theta)
      #    x0 = a*rho
      #    y0 = b*rho
      #    x1 = int(x0 + 5*(-b))
      #    y1 = int(y0 + 5*(a))
      #    x2 = int(x0 - 5*(-b))
      #    y2 = int(y0 - 5*(a))
      #    cv2.line(roi_line,(x1,y1),(x2,y2),(0,0,255),2)
      #    break
      for i in range(0,1):
        rho,theta = lines[0][i]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10*(-b))
        y1 = int(y0 + 10*(a))
        x2 = int(x0 - 10*(-b))
        y2 = int(y0 - 10*(a))
        cv2.line(roi_line,(x1,y1),(x2,y2),(0,0,255),2)

    img_lines[block_x*9:block_x*9+8,block_y*9:block_y*9+8] = roi_line
    img[block_x*9:block_x*9+8,block_y*9:block_y*9+8] = roi
    #img_blur[block_x*5:block_x*5+4,block_y*5:block_y*5+4] = edges


plt.subplot(221),plt.imshow(img_ori,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.subplot(222),plt.imshow(img_blur,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_dog,cmap = 'gray')
plt.title('DOG Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img_lines,cmap = 'gray')
plt.title('Line Image'), plt.xticks([]), plt.yticks([])

#today=datetime.now().strftime("%y-%m-%d_%H-%M_")
#np.savetxt(today+"img.csv", img, '%.u', ',')
#img_lines_gray = cv2.cvtColor(img_lines,cv2.COLOR_BGR2GRAY)
#np.savetxt(today+"img_lines.csv", img_lines_gray, '%.u', ',')
#img_lines_all_gray = cv2.cvtColor(img_lines_all,cv2.COLOR_BGR2GRAY)
#np.savetxt(today+"img_lines_all.csv", img_lines_all_gray, '%.u', ',')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

