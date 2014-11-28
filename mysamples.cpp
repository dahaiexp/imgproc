# Computing the HOG descriptor using OpenCV

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
Mat img_raw = imread("data\\pedestrians128x64\\per00001.ppm", 1); // load as color image
 
resize(img_raw, img_raw, Size(64,128) );
 
Mat img;
cvtColor(img_raw, img, CV_RGB2GRAY);
 
 
HOGDescriptor d;
// Size(128,64), //winSize
// Size(16,16), //blocksize
// Size(8,8), //blockStride,
// Size(8,8), //cellSize,
// 9, //nbins,
// 0, //derivAper,
// -1, //winSigma,
// 0, //histogramNormType,
// 0.2, //L2HysThresh,
// 0 //gammal correction,
// //nlevels=64
//);
 
// void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
//                             Size winStride, Size padding,
//                             const vector<Point>& locations) const
vector<float> descriptorsValues;
vector<Point> locations;
d.compute( img, descriptorsValues, Size(0,0), Size(0,0), locations);
 
cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
cout << "Nr of locations specified : " << locations.size() << endl;

    return 0;
}
