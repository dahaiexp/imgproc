import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt
from math import sqrt
from math import tan
from math import radians
import time
from datetime import date
from datetime import datetime

def height(img):
  return img.shape[1]

def width(img):
  return img.shape[0]

def is_close(x1, x2, dx):
  if x1>x2: return (x1-x2<dx)
  else: return (x2-x1<dx)

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
    if(is_close( dist(y,x,circle_center_x(cir),circle_center_y(cir)), circle_radius(cir), 1)): return True
    #if(first): break
  return False

#######################################################
# 
def left(img, step):
  y=0
  while y < height(img):
      yield 0,y
      y += step

def top(img, step):
  x=0
  while x < width(img):
      yield x,0
      x += step

def right(img, step):
  y=0
  while y < height(img):
      yield width(img)-1,y
      y += step

def bottom(img, step):
  x=0
  while x < width(img):
      yield x, height(img)-1
      x += step

def scan(img, step):
  x=0;y=0
  while y < height(img) and x < width(img):
      yield x, y
      x += step
      if (x==width(img)):
        x=0;y+=step

def draw_plane(img_out, a, b, d, z1):
  for x,y in scan(img_out, 5):
    value=-a*x-b*y-d+z1
    #print "x=", x, ";y=", y, ";value=%.2f" % value
    if (value>255): value=255
    img_out[x,y]=value

visited={}
def visited_4f(a,b,d,z1):
  key="{:.1f}".format(a)+'_'+"{:.1f}".format(b)+'_'+"{:.1f}".format(d)+'_'+"{:.1f}".format(z1)
  if (key in visited): return True;
  visited[key]=True

visited2={}
def visited_5i(x1,y1,x2,y2,z1):
  key=str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2)+'_'+str(z1)
  if (key in visited2): return True;
  visited2[key]=True

def generate_plane_theta(img, x1, y1, x2, y2, z1, theta):
  # special cases
  print "x1=", x1,"; y1=", y1,"; x2=", x2,"; y2=", y2,"; theta=", theta, ";z1=", z1

  coords = []
  if (theta==0): a=0; b=0; d=0; coords.append([a,b,d])
  elif (x1==x2):
    b=0; a=1.0*tan(radians(theta)); d=-float(a)*x1; coords.append([a,b,d])
    b=-b; a=-a; d=-d
    coords.append([a,b,d])
  elif (y1==y2):
    a=0; b=1.0*tan(radians(theta)); d=-float(b)*y1; coords.append([a,b,d])
    b=-b; a=-a; d=-d
    coords.append([a,b,d])
  else :
    b=float(x1-x2)/sqrt((x1-x2)**2+(y1-y2)**2)*tan(radians(theta))
    a=-b*(y1-y2)/(x1-x2)
    d=-a*x1-b*y1
    coords.append([a,b,d])
    b=-b; a=-a; d=-d
    coords.append([a,b,d])

  #print coords
  for (a1,b1,d1) in coords:
    if not visited_4f(a1,b1,d1,z1):
      #print "a=%f;b=%f;d=%f" % (a1, b1, d1)
      draw_plane(img, a1, b1, d1, z1)
      #cv2.imwrite('gen_plane_out_'+"{:.1f}".format(a1)+'_'+"{:.1f}".format(b)+'_'+"{:.1f}".format(d1)+'_'+"{:.1f}".format(z1)+'.png', img)
      str=np.array_str(img)
      output_file.write("---------------------------------\n")
      #output_file.write("x1:%d"+str(x1)+";y1:"+str(y1)+";x2="+str(x2)+";y2="+str(y2)+";z1="+str(z1)+";theta="+str(theta)+"\n")
      output_file.write("x1=%d;y1=%d;x2=%d;y2=%d;z1=%d;theta=%d;\n" % (x1, y1, x2, y2, z1, theta) )
      output_file.write("a=%.1f;b=%.1f;d=%.1f;theta=%d\n" % (a1, b1, d1, theta) )
      output_file.write(str)
      output_file.write("---------------------------------\n")


def generate_plane(img, x1, y1, x2, y2, z1):
  # a*x+b*y+(z-z1)+d=0
  if (x1==x2 and y1==y2): return;
  if visited_5i(x1,y1,x2,y2,z1): return;

  #max_size=max(height(img), width(img))
  #angle_step=degrees(atan(1.0/max_size))
  #if(angle_step<1): angle_step=1
  #print "max_size:%d; angle_step:%d" % (max_size, angle_step)

  angle_step=10

  for theta in range(0,40,angle_step):
    generate_plane_theta(img, x1, y1, x2, y2, z1, theta)


#######################################################
# 
blank_image = np.zeros((30,40), np.uint8) #(height,width)
output_file= open("auto_planes_log.txt", "w")

#generate_plane_theta(blank_image, 0, 0, 1, 0, 0, 45)

for z in range(0,255,50):
  print "start:"
  for x1,y1 in left(blank_image, 10):
    print "---------------------"
    print "x1=", x1, "; y1=", y1,";z1=", z
    for x2,y2 in top(blank_image, 10): generate_plane(blank_image, x1,y1,x2,y2,z)
    print "---------------------"
    print "x1=", x1, "; y1=", y1,";z1=", z
    for x2,y2 in right(blank_image, 10): generate_plane(blank_image, x1,y1,x2,y2,z)
    print "---------------------"
    print "x1=", x1, "; y1=", y1,";z1=", z
    for x2,y2 in bottom(blank_image, 10): generate_plane(blank_image, x1,y1,x2,y2,z)
    print "---------------------"
  for x1,y1 in top(blank_image, 10):
    print "---------------------"
    print "x1=", x1, "; y1=", y1,";z1=", z
    for x2,y2 in right(blank_image, 10): generate_plane(blank_image, x1,y1,x2,y2,z)
    print "---------------------"
    print "x1=", x1, "; y1=", y1,";z1=", z
    for x2,y2 in bottom(blank_image, 10): generate_plane(blank_image, x1,y1,x2,y2,z)
    print "---------------------"
  for x1,y1 in right(blank_image, 10):
    print "---------------------"
    print "x1=", x1, "; y1=", y1,";z1=", z
    for x2,y2 in bottom(blank_image, 10): generate_plane(blank_image, x1,y1,x2,y2,z)
    print "---------------------"

output_file.close()

#plt.show()
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()

