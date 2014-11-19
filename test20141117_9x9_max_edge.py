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

g_count=0

def find_edge(gray_block, block_size, mask_size, thres):
  global g_count

  # initialize
  max_degree=-1
  max_pos=-1
  max_sum=-1

  # calculate DoG 
  gray_blur7 = cv2.medianBlur(gray_block,7)
  #gray_dog = cv2.subtract(gray_block, gray_blur7)
  gray_dog = cv2.subtract(gray_blur7, gray_block)

  # for each degree, get max sum
  for degree in range(0, 180, 10):
    #print "degree:"+str(degree)

    # rotate the image, use a mask to filter out the needed part
    M = cv2.getRotationMatrix2D((block_size/2,block_size/2),degree,1)
    dst = cv2.warpAffine(gray_dog,M,(block_size,block_size))[block_size/2-mask_size/2:block_size/2+mask_size/2,block_size/2-mask_size/2:block_size/2+mask_size/2]
    #cv2.imwrite('building_debug'+str(g_count)+'.png', dst)
    g_count=g_count+1
    pos = (degree/10)+1
    if (pos >= 10): pos=pos+9
    #plt.subplot(4, 9, pos),plt.imshow(dst,cmap = 'gray')
    #sum_dst = np.sum(dst,axis=0)
    sum_dst = np.median(dst,axis=0)
  
    sum_dst_max_idx = np.argmax(sum_dst)
    #print "argmax:"+str(sum_dst_max_idx)
    sum_dst_max=sum_dst[sum_dst_max_idx]
    #print "sum_dst_max:"+str(sum_dst_max)
    if (sum_dst_max > max_sum):
      #print "new_max_degree"
      max_degree=degree
      max_pos=sum_dst_max_idx
      max_sum = sum_dst_max
  
    #plt.subplot(4, 9, pos+9),plt.plot(sum_dst)
    #plt.subplot(2, 18, (degree/10)+1+18),plt.hist(dst.ravel(),256,[0,256])
  
  gray_block=cv2.cvtColor(gray_block[block_size/2-mask_size/2:block_size/2+mask_size/2,block_size/2-mask_size/2:block_size/2+mask_size/2],cv2.COLOR_GRAY2BGR)
  if (max_sum < thres): return gray_block;

  dist_from_center=max_pos-mask_size/2
  print "dist_from_center:"+str(dist_from_center)+";block_size:"+str(block_size)
  theta=(max_degree+0)/180.0*np.pi
  print "theta:"+str(theta)+";max_degree:"+str(max_degree)
  rho=dist_from_center
  a = np.cos(theta)
  b = np.sin(theta)
  print "a:"+str(theta)+";b:"+str(b)
  x0 = a*rho+mask_size/2
  y0 = b*rho+mask_size/2
  print "x0:"+str(x0)+";y0:"+str(y0)
  x1 = int(x0 + 100*(-b))
  y1 = int(y0 + 100*(a))
  x2 = int(x0 - 100*(-b))
  y2 = int(y0 - 100*(a))
  print "x1:"+str(x1)+";y1:"+str(y1)
  print "x2:"+str(x2)+";y2:"+str(y2)
  cv2.line(gray_block,(x1,y1),(x2,y2),(0,0,255),2)
  #cv2.line(gray_dog,(0,0),(140,140),(0,0,255),2)
  #plt.subplot(4, 9, 1),plt.imshow(gray_dog,cmap = 'gray')
  return gray_block

#######################################################
# 1. for each degree, rotate pixel
# 2. calculate histogram
# 3. find the max, remember the degree and position.
# 4. display an edge along the max position and degree.

# 1. for each degree, rotate pixel
img = cv2.imread('building.jpg')
img_out=img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rows,cols = gray.shape
# calculate DoG 
gray_blur7 = cv2.medianBlur(gray,7)
#gray_dog = cv2.subtract(gray, gray_blur7)
gray_dog = cv2.subtract(gray_blur7, gray)

gray_out=find_edge(gray, rows, rows/1.5, 10)
cv2.imwrite("building-out.png", gray_out)
#plt.subplot(1, 1, 1),plt.imshow(gray_out,cmap = 'gray')
#

############################################################
##img = cv2.imread('building.jpg')
##img_out=img.copy()
##gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##rows,cols = gray.shape
### calculate DoG 
##gray_blur7 = cv2.medianBlur(gray,7)
###gray_dog = cv2.subtract(gray, gray_blur7)
##gray_dog = cv2.subtract(gray_blur7, gray)
##block_size = 16
##mask_size = 10
###block_num = rows / block_size
##
###
##for left in range(0, cols-block_size, mask_size):
##  for top in range(0, rows-block_size, mask_size):
##    #if True:
##    #left=50
##    #top=50
##    roi=gray[left:left+block_size, top:top+block_size]
##    gray_out=find_edge(roi, block_size, mask_size, 10)
##    cv2.imwrite('building_in_'+str(left)+'_'+str(top)+'.png', roi)
##    cv2.imwrite('building_out_'+str(left)+'_'+str(top)+'.png', gray_out)
##    img_out[left+block_size/2-mask_size/2:left+block_size/2+mask_size/2, top+block_size/2-mask_size/2:top+block_size/2+mask_size/2]=gray_out
##
##fig=plt.figure()
##ax1=fig.add_subplot(121)
##ax1.imshow(gray_dog)
##for left in range(0, cols-block_size, mask_size):
##  ax1.plot([left+block_size/2-mask_size/2-0.5, left+block_size/2-mask_size/2-0.5], [1-0.5, rows-0.5], 'k-', linewidth=1)
##
##for top in range(0, rows-block_size, mask_size):
##  ax1.plot([1-0.5, cols-0.5], [top+block_size/2-mask_size/2-0.5, top+block_size/2-mask_size/2-0.5], 'k-', linewidth=1)
##
##ax1.axis([0,cols,rows, 0])
##
##ax2=fig.add_subplot(122)
###plt.subplot(122),
##ax2.imshow(img_out)
##
##for left in range(0, cols-block_size, mask_size):
##  ax2.plot([left+block_size/2-mask_size/2-0.5, left+block_size/2-mask_size/2-0.5], [1-0.5, rows-0.5], 'k-', linewidth=1)
##
##for top in range(0, rows-block_size, mask_size):
##  ax2.plot([1-0.5, cols-0.5], [top+block_size/2-mask_size/2-0.5, top+block_size/2-mask_size/2-0.5], 'k-', linewidth=1)
##
##ax2.axis([0,cols,rows, 0])

#rows,cols = gray.shape
#rows,cols = gray.shape
#if (rows > cols): rows=cols
#
#max_degree=-1
#max_pos=-1
#max_sum=-1
#gray_blur7 = cv2.medianBlur(gray,7)
#gray_dog = cv2.subtract(gray, gray_blur7)
#for degree in range(0, 180, 10):
#  print "degree:"+str(degree)
#  M = cv2.getRotationMatrix2D((rows/2,rows/2),degree,1)
#  dst = cv2.warpAffine(gray_dog,M,(rows,rows))[rows/2-30:rows/2+30,rows/2-30:rows/2+30]
#  pos = (degree/10)+1
#  if (pos >= 10): pos=pos+9
#  plt.subplot(4, 9, pos),plt.imshow(dst,cmap = 'gray')
#  sum_dst = np.sum(dst,axis=0)
#
#  sum_dst_max_idx = np.argmax(sum_dst)
#  print "argmax:"+str(sum_dst_max_idx)
#  sum_dst_max=sum_dst[sum_dst_max_idx]
#  print "sum_dst_max:"+str(sum_dst_max)
#  if (sum_dst_max > max_sum):
#    print "new_max_degree"
#    max_degree=degree
#    max_pos=sum_dst_max_idx
#    max_sum = sum_dst_max
#
#  plt.subplot(4, 9, pos+9),plt.plot(sum_dst)
#  #plt.subplot(2, 18, (degree/10)+1+18),plt.hist(dst.ravel(),256,[0,256])
#
#dist_from_center=max_pos-30
#print "dist_from_center:"+str(dist_from_center)+";rows:"+str(rows)
#theta=(max_degree+0)/180.0*np.pi
#print "theta:"+str(theta)+";max_degree:"+str(max_degree)
#rho=dist_from_center
#a = np.cos(theta)
#b = np.sin(theta)
#print "a:"+str(theta)+";b:"+str(b)
#x0 = a*rho+30
#y0 = b*rho+30
#print "x0:"+str(x0)+";y0:"+str(y0)
#x1 = int(x0 + 100*(-b))
#y1 = int(y0 + 100*(a))
#x2 = int(x0 - 100*(-b))
#y2 = int(y0 - 100*(a))
#print "x1:"+str(x1)+";y1:"+str(y1)
#print "x2:"+str(x2)+";y2:"+str(y2)
#gray_dog=cv2.cvtColor(gray_dog[rows/2-30:rows/2+30,rows/2-30:rows/2+30],cv2.COLOR_GRAY2BGR)
#cv2.line(gray_dog,(x1,y1),(x2,y2),(0,0,255),2)
#cv2.line(gray_dog,(0,0),(140,140),(0,0,255),2)
#plt.subplot(4, 9, 1),plt.imshow(gray_dog,cmap = 'gray')
# 2. calculate histogram
# 3. find the max, remember the degree and position.
# 4. display an edge along the max position and degree.

#plt.subplot(381),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
#plt.subplot(382),plt.hist(img.ravel(),256,[0,256]);
#
#plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(383),plt.imshow(dst,cmap = 'gray')
#plt.title('DOG Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(384),plt.hist(dst.ravel(),256,[0,256]);
#plt.title('Line Image'), plt.xticks([]), plt.yticks([])

#plt.show()
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()

########################################################
##img = cv2.imread('building.jpg')
##gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##edges = cv2.Canny(gray,50,150,apertureSize = 3)
##
##lines = cv2.HoughLines(edges,1,np.pi/180,50)
##if (lines is not None):
##  count = 0
##  for rho,theta in lines[0]:
##      a = np.cos(theta)
##      b = np.sin(theta)
##      x0 = a*rho
##      y0 = b*rho
##      x1 = int(x0 + 1000*(-b))
##      y1 = int(y0 + 1000*(a))
##      x2 = int(x0 - 1000*(-b))
##      y2 = int(y0 - 1000*(a))
##
##      cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
##      count += 1
##      if (count>10): break
##
##plt.subplot(111),plt.imshow(img,cmap = 'gray')
##plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##
##plt.show()
##cv2.waitKey(0)
##cv2.destroyAllWindows()
########################################################
#
##########################################################
##img = cv2.imread('building.jpg')
##gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##gray_blur3 = cv2.medianBlur(gray,3)
##gray_blur7 = cv2.medianBlur(gray,7)
##gray_dog = cv2.subtract(gray, gray_blur7)
##gray_dog_out = gray_dog.copy()
##gray_dog_out = cv2.cvtColor(gray_dog_out,cv2.COLOR_GRAY2BGR)
###edges = cv2.Canny(gray,50,150,apertureSize = 3)
##
###lines = cv2.HoughLines(gray_dog,10,np.pi/18,300)
##lines = cv2.HoughLines(gray_dog,1,np.pi/180,10)
##if (lines is not None):
##  print lines[0]
##  for i in range(0,10):
##    if (lines[0][i] is not None):
##          rho,theta = lines[0][i]
##          a = np.cos(theta)
##          b = np.sin(theta)
##          x0 = a*rho
##          y0 = b*rho
##          x1 = int(x0 + 1000*(-b))
##          y1 = int(y0 + 1000*(a))
##          x2 = int(x0 - 1000*(-b))
##          y2 = int(y0 - 1000*(a))
##          cv2.line(gray_dog_out,(x1,y1),(x2,y2),(0,0,255),2)
##
##plt.subplot(221),plt.imshow(gray,cmap = 'gray')
##plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##plt.subplot(222),plt.imshow(gray_dog,cmap = 'gray')
##plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##plt.subplot(223),plt.imshow(gray_dog,cmap = 'gray')
##plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##plt.subplot(224),plt.imshow(gray_dog_out,cmap = 'gray')
##plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##
##plt.show()
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##########################################################
#
## Load an color image in grayscale
#img_ori = cv2.imread('building.jpg')
#img = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
#img_blur = cv2.medianBlur(img,7)
##img_dog=img-img_blur
#img_dog = cv2.subtract(img, img_blur)
#img_lines = img_ori.copy()
#
#edges = cv2.Canny(img,50,150,apertureSize = 3)
##lines_all = cv2.HoughLines(edges,1,np.pi/180,200)
#img_lines_all = img_ori.copy()
##if (lines_all is not None):
##  for rho,theta in lines_all[0]:
##      a = np.cos(theta)
##      b = np.sin(theta)
##      x0 = a*rho
##      y0 = b*rho
##      x1 = int(x0 + 5*(-b))
##      y1 = int(y0 + 5*(a))
##      x2 = int(x0 - 5*(-b))
##      y2 = int(y0 - 5*(a))
##      cv2.line(img_lines_all,(x1,y1),(x2,y2),(0,0,255),2)
##      break
#
## for each 5x5 block, calculate hough line
#for block_y in range(0,width(img)/9):
#  for block_x in range(0,height(img)/9):
#    roi = img_dog[block_x*9:block_x*9+8,block_y*9:block_y*9+8]
#    roi_line = img_ori[block_x*9:block_x*9+8,block_y*9:block_y*9+8]
#    #edges = cv2.Canny(roi,50,150,apertureSize = 3)
#    lines = cv2.HoughLines(roi,1,np.pi/180,10)
#    #if(lines is not None and lines.any()):
#    if (lines is not None):
#      #for rho,theta in lines[0]:
#      #    a = np.cos(theta)
#      #    b = np.sin(theta)
#      #    x0 = a*rho
#      #    y0 = b*rho
#      #    x1 = int(x0 + 5*(-b))
#      #    y1 = int(y0 + 5*(a))
#      #    x2 = int(x0 - 5*(-b))
#      #    y2 = int(y0 - 5*(a))
#      #    cv2.line(roi_line,(x1,y1),(x2,y2),(0,0,255),2)
#      #    break
#      for i in range(0,1):
#        rho,theta = lines[0][i]
#        a = np.cos(theta)
#        b = np.sin(theta)
#        x0 = a*rho
#        y0 = b*rho
#        x1 = int(x0 + 10*(-b))
#        y1 = int(y0 + 10*(a))
#        x2 = int(x0 - 10*(-b))
#        y2 = int(y0 - 10*(a))
#        cv2.line(roi_line,(x1,y1),(x2,y2),(0,0,255),2)
#
#    img_lines[block_x*9:block_x*9+8,block_y*9:block_y*9+8] = roi_line
#    img[block_x*9:block_x*9+8,block_y*9:block_y*9+8] = roi
#    #img_blur[block_x*5:block_x*5+4,block_y*5:block_y*5+4] = edges
#
#
#plt.subplot(221),plt.imshow(img_ori,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
#
#plt.subplot(222),plt.imshow(img_blur,cmap = 'gray')
#plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(223),plt.imshow(img_dog,cmap = 'gray')
#plt.title('DOG Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(224),plt.imshow(img_lines,cmap = 'gray')
#plt.title('Line Image'), plt.xticks([]), plt.yticks([])
#
##today=datetime.now().strftime("%y-%m-%d_%H-%M_")
##np.savetxt(today+"img.csv", img, '%.u', ',')
##img_lines_gray = cv2.cvtColor(img_lines,cv2.COLOR_BGR2GRAY)
##np.savetxt(today+"img_lines.csv", img_lines_gray, '%.u', ',')
##img_lines_all_gray = cv2.cvtColor(img_lines_all,cv2.COLOR_BGR2GRAY)
##np.savetxt(today+"img_lines_all.csv", img_lines_all_gray, '%.u', ',')
#
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

