//  #include "opencv2/imgproc/imgproc.hpp"
//  #include "opencv2/contrib/contrib.hpp"
//  #include "opencv2/highgui/highgui.hpp"
//  
//  #include <cstdio>
//  #include <iostream>
//  #include <ctime>
//  
//  using namespace cv;
//  using namespace std;
//  
//  
//  int main(int argc, char** argv)
//  {
//  Mat img_raw = imread(argv[1], 1); // load as color image
//   
//  resize(img_raw, img_raw, Size(64,128) );
//   
//  Mat img;
//  cvtColor(img_raw, img, CV_RGB2GRAY);
//   
//   
//  HOGDescriptor d;
//  // Size(128,64), //winSize
//  // Size(16,16), //blocksize
//  // Size(8,8), //blockStride,
//  // Size(8,8), //cellSize,
//  // 9, //nbins,
//  // 0, //derivAper,
//  // -1, //winSigma,
//  // 0, //histogramNormType,
//  // 0.2, //L2HysThresh,
//  // 0 //gammal correction,
//  // //nlevels=64
//  //);
//   
//  // void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
//  //                             Size winStride, Size padding,
//  //                             const vector<Point>& locations) const
//  vector<float> descriptorsValues;
//  vector<Point> locations;
//  d.compute( img, descriptorsValues, Size(0,0), Size(0,0), locations);
//   
//  cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
//  cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
//  cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
//  cout << "Nr of locations specified : " << locations.size() << endl;
//  
//  
//  // HOGDescriptor visual_imagealizer
//  // adapted for arbitrary size of feature sets and training images
//  //Mat get_hogdescriptor_visual_image(Mat& origImg,
//  //                                   vector<float>& descriptorValues,
//  //                                   Size winSize,
//  //                                   Size cellSize,                                   
//  //                                   int scaleFactor,
//  //                                   double viz_factor)
//  Mat mat_result;
//  Mat& origImg=img;
//  vector<float>& descriptorValues=descriptorsValues;
//  Size winSize=Size(0,0);
//  Size cellSize=Size(0,0);
//  int scaleFactor=1;
//  double viz_factor=1.0;
//  {   
//  cout << "1;";
//      Mat visual_image;
//      resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
//   
//  cout << "2;";
//      int gradientBinSize = 9;
//      // dividing 180° into 9 bins, how large (in rad) is one bin?
//      float radRangeForOneBin = 3.14/(float)gradientBinSize; 
//   
//      // prepare data structure: 9 orientation / gradient strenghts for each cell
//  	int cells_in_x_dir = winSize.width / cellSize.width;
//      int cells_in_y_dir = winSize.height / cellSize.height;
//      int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
//      float*** gradientStrengths = new float**[cells_in_y_dir];
//      int** cellUpdateCounter   = new int*[cells_in_y_dir];
//  cout << "3;";
//      for (int y=0; y<cells_in_y_dir; y++)
//      {
//          gradientStrengths[y] = new float*[cells_in_x_dir];
//          cellUpdateCounter[y] = new int[cells_in_x_dir];
//          for (int x=0; x<cells_in_x_dir; x++)
//          {
//              gradientStrengths[y][x] = new float[gradientBinSize];
//              cellUpdateCounter[y][x] = 0;
//   
//              for (int bin=0; bin<gradientBinSize; bin++)
//                  gradientStrengths[y][x][bin] = 0.0;
//          }
//      }
//   
//  cout << "4;";
//      // nr of blocks = nr of cells - 1
//      // since there is a new block on each cell (overlapping blocks!) but the last one
//      int blocks_in_x_dir = cells_in_x_dir - 1;
//      int blocks_in_y_dir = cells_in_y_dir - 1;
//   
//      // compute gradient strengths per cell
//      int descriptorDataIdx = 0;
//      int cellx = 0;
//      int celly = 0;
//   
//      for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
//      {
//          for (int blocky=0; blocky<blocks_in_y_dir; blocky++)            
//          {
//              // 4 cells per block ...
//              for (int cellNr=0; cellNr<4; cellNr++)
//              {
//                  // compute corresponding cell nr
//                  int cellx = blockx;
//                  int celly = blocky;
//                  if (cellNr==1) celly++;
//                  if (cellNr==2) cellx++;
//                  if (cellNr==3)
//                  {
//                      cellx++;
//                      celly++;
//                  }
//   
//                  for (int bin=0; bin<gradientBinSize; bin++)
//                  {
//                      float gradientStrength = descriptorValues[ descriptorDataIdx ];
//                      descriptorDataIdx++;
//   
//                      gradientStrengths[celly][cellx][bin] += gradientStrength;
//   
//                  } // for (all bins)
//   
//   
//                  // note: overlapping blocks lead to multiple updates of this sum!
//                  // we therefore keep track how often a cell was updated,
//                  // to compute average gradient strengths
//                  cellUpdateCounter[celly][cellx]++;
//   
//              } // for (all cells)
//   
//   
//          } // for (all block x pos)
//      } // for (all block y pos)
//   
//  cout << "5;";
//   
//      // compute average gradient strengths
//      for (int celly=0; celly<cells_in_y_dir; celly++)
//      {
//          for (int cellx=0; cellx<cells_in_x_dir; cellx++)
//          {
//   
//              float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
//   
//              // compute average gradient strenghts for each gradient bin direction
//              for (int bin=0; bin<gradientBinSize; bin++)
//              {
//                  gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
//              }
//          }
//      }
//   
//   
//      cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
//  cout << "6;";
//   
//      // draw cells
//      for (int celly=0; celly<cells_in_y_dir; celly++)
//      {
//          for (int cellx=0; cellx<cells_in_x_dir; cellx++)
//          {
//              int drawX = cellx * cellSize.width;
//              int drawY = celly * cellSize.height;
//   
//              int mx = drawX + cellSize.width/2;
//              int my = drawY + cellSize.height/2;
//   
//              rectangle(visual_image,
//                        Point(drawX*scaleFactor,drawY*scaleFactor),
//                        Point((drawX+cellSize.width)*scaleFactor,
//                        (drawY+cellSize.height)*scaleFactor),
//                        CV_RGB(100,100,100),
//                        1);
//   
//              // draw in each cell all 9 gradient strengths
//              for (int bin=0; bin<gradientBinSize; bin++)
//              {
//                  float currentGradStrength = gradientStrengths[celly][cellx][bin];
//   
//                  // no line to draw?
//                  if (currentGradStrength==0)
//                      continue;
//   
//                  float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
//   
//                  float dirVecX = cos( currRad );
//                  float dirVecY = sin( currRad );
//                  float maxVecLen = cellSize.width/2;
//                  float scale = viz_factor; // just a visual_imagealization scale,
//                                            // to see the lines better
//   
//                  // compute line coordinates
//                  float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
//                  float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
//                  float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
//                  float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
//   
//                  // draw gradient visual_imagealization
//                  line(visual_image,
//                       Point(x1*scaleFactor,y1*scaleFactor),
//                       Point(x2*scaleFactor,y2*scaleFactor),
//                       CV_RGB(0,0,255),
//                       1);
//   
//              } // for (all bins)
//   
//          } // for (cellx)
//      } // for (celly)
//   
//   
//  cout << "7;";
//      // don't forget to free memory allocated by helper data structures!
//      for (int y=0; y<cells_in_y_dir; y++)
//      {
//        for (int x=0; x<cells_in_x_dir; x++)
//        {
//             delete[] gradientStrengths[y][x];            
//        }
//        delete[] gradientStrengths[y];
//        delete[] cellUpdateCounter[y];
//      }
//      delete[] gradientStrengths;
//      delete[] cellUpdateCounter;
//   
//  cout << "8;";
//  //    return visual_image;
//  mat_result=visual_image;
//   
//  }
//  
//  // output result
//  //Mat get_hogdescriptor_visual_image(Mat& origImg,
//  //                                   vector<float>& descriptorValues,
//  //                                   Size winSize,
//  //                                   Size cellSize,                                   
//  //                                   int scaleFactor,
//  //                                   double viz_factor)
//  
//  
//  cout << "Visualize : " << endl;
//  //Mat mat_result = get_hogdescriptor_visual_image(img, descriptorsValues, Size(0,0), Size(0,0), 1, 1);
//    
//    cout << "Output file:" << mat_result.cols << " width x " << mat_result.rows << "height" << endl;
//  imwrite("per00001-out.png", mat_result);
//      return 0;
//  }


////
////#include <opencv2/imgproc/imgproc.hpp>
////#include <opencv2/objdetect/objdetect.hpp>
////#include <opencv2/highgui/highgui.hpp>
////
////#include <stdio.h>
////#include <string.h>
////#include <ctype.h>
////
////using namespace cv;
////using namespace std;
////
////// static void help()
////// {
//////     printf(
//////             "\nDemonstrate the use of the HoG descriptor using\n"
//////             "  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"
//////             "Usage:\n"
//////             "./peopledetect (<image_filename> | <image_list>.txt)\n\n");
////// }
////
////int main(int argc, char** argv)
////{
////    Mat img;
////    FILE* f = 0;
////    char _filename[1024];
////
////    if( argc == 1 )
////    {
////        printf("Usage: peopledetect (<image_filename> | <image_list>.txt)\n");
////        return 0;
////    }
////    img = imread(argv[1]);
////
////    if( img.data )
////    {
////        strcpy(_filename, argv[1]);
////    }
////    else
////    {
////        f = fopen(argv[1], "rt");
////        if(!f)
////        {
////            fprintf( stderr, "ERROR: the specified file could not be loaded\n");
////            return -1;
////        }
////    }
////
////    HOGDescriptor hog;
////    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());   //设置线性SVM分类器的系数
////    namedWindow("people detector", 1);                               //新建一个窗口
////
////    for(;;)
////    {
////        char* filename = _filename;
////        if(f)
////        {
////            if(!fgets(filename, (int)sizeof(_filename)-2, f))
////                break;
////            //while(*filename && isspace(*filename))
////            //  ++filename;
////            if(filename[0] == '#')
////                continue;
////            int l = (int)strlen(filename);
////            while(l > 0 && isspace(filename[l-1]))
////                --l;
////            filename[l] = '\0';
////            img = imread(filename);
////        }
////        printf("%s:\n", filename);
////        if(!img.data)
////            continue;
////
////        fflush(stdout);
////        vector<Rect> found, found_filtered;
////        double t = (double)getTickCount();
////        // run the detector with default parameters. to get a higher hit-rate
////        // (and more false alarms, respectively), decrease the hitThreshold and
////        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
////        hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
////        t = (double)getTickCount() - t;
////        printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
////        size_t i, j;
////        for( i = 0; i < found.size(); i++ )
////        {
////            Rect r = found[i];
////
////			//下面的这个for语句是找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的
////			//话,则取外面最大的那个矩形框放入found_filtered中
////            for( j = 0; j < found.size(); j++ )
////                if( j != i && (r & found[j]) == r)
////                    break;                         //如果r被其他矩形包含，则丢弃r
////
////            if( j == found.size() )                //如果r不被其他矩形包含，则保留r
////                found_filtered.push_back(r);
////        }
////        for( i = 0; i < found_filtered.size(); i++ )
////        {
////            Rect r = found_filtered[i];
////            // the HOG detector returns slightly larger rectangles than the real objects.
////            // so we slightly shrink the rectangles to get a nicer output.
////            r.x += cvRound(r.width*0.1);
////            r.width = cvRound(r.width*0.8);
////            r.y += cvRound(r.height*0.07);
////            r.height = cvRound(r.height*0.8);
////            rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
////        }
////        imshow("people detector", img);
////        int c = waitKey(0) & 255;
////        if( c == 'q' || c == 'Q' || !f)
////            break;
////    }
////    if(f)
////        fclose(f);
////    return 0;
////}


#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
cv::Mat img = cv::imread( argv[1] );                                     // 入力画像
cv::HOGDescriptor hog;                                                          //
hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());              // 
std::vector<cv::Rect> found;                                                    // 検出結果格納用vector
cv::Mat window;
resize(img, window, Size(64,128) );
//hog.detect(window, found);
hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(16,16), 1.05, 20);   // パラメータ設定
                                                                                // 画像，検出結果，閾値（SVMのhyper-planeとの距離），
                                                                                // 探索窓の移動距離（Block移動距離の倍数），
                                                                                // 画像外にはみ出た対象を探すためのpadding，
                                                                                // 探索窓のスケール変化係数，グルーピング係数

cout << "finished:" << found.size() << endl;
std::vector<cv::Rect>::const_iterator it = found.begin();                       // 検出結果の矩形を画像に書き込む
// std::cout << "found:" << found.size() << std::endl;
for(; it!=found.end(); ++it) {
    cout << "found:" << endl;
    cv::Rect r = *it;
    cv::rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
}
imwrite("per00001-out.png", img);
}
