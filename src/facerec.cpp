#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utils/logger.hpp" //new
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

// g++ -std=c++17 facerec.cpp -lopencv_face -lopencv_core -lopencv_imgcodecs
int g_redness = 255;

int main(int argc, char *argv[])
{
  namespace fs = std::filesystem;

  std::vector<cv::Mat> images;
  std::vector<int>     labels;

  // Iterate through all subdirectories, looking for .pgm files
  fs::path p(argc > 1 ? argv[1] : "../../att_faces");
  for (const auto &entry : fs::recursive_directory_iterator{ p }) {
    if (fs::is_regular_file(entry.status())) { // Was once always (wrongly) false in VS
      if (entry.path().extension() == ".pgm") {
        std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
        int label = atoi(str.c_str() + 1); // s1 -> 1 (pointer arithmetic)
        images.push_back(cv::imread(entry.path().string().c_str(), cv::IMREAD_GRAYSCALE));
        labels.push_back(label);
      }
    }
  }

  std::cout << " training...\n";
  cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->train(images, labels);

  /*
  int predictedLabel = model->predict(testSample);
  std::cout << "\nPredicted class = " << predictedLabel << '\n';*/

  //new - this has been taken from 04_capture_show_video.cpp from the open cv lab
  cv::Mat frame;
  double fps = 30;
  const char win_name[] = "Live Video...";

  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

  std::cout << "Wait 60 secs. for camera access to be obtained..." << std::endl;
  cv::VideoCapture vid_in(0);   // argument is the camera id

  if (vid_in.isOpened())
  {
      std::cout << "Camera capture obtained." << std::endl;
  }
  else
  {
      std::cerr << "error: Camera 0 could not be opened for capture.\n";
      return -1;
  }

  cv::namedWindow(win_name);
  

  cv::Point center(320, 240); //centre of screen
  cv::Point p1 (274, 296); //bottom left
  cv::Point p2 (366, 184); // top right
  int i{ 0 };

  while (1) {
      vid_in >> frame;
      
      cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY); //makes webcam greyscale

      cv::Rect box; //creates rectangle object
      box.x = 228, box.y = 128, box.width = 92*2, box.height = 112*2; //defines box params
      cv::Mat srcBox = frame(box); //create mat based on box
      cv::Canny(srcBox, srcBox, 120, 150); //canny effect inside box

      //effects
      /*cv::GaussianBlur(srcBox, srcBox, cv::Size(13, 13), 5, 5); //creates Gaussian Blur only inside box
      cv::Mat corners;
      cv::goodFeaturesToTrack(frame, corners, 10, 0.01, 50); //not sure this works yet */

      imshow(win_name, frame);
      
      int code = cv::waitKey(1000 / fps); // how long to wait for a key (msecs)
      if (code == 27) // escape. See http://www.asciitable.com/
          break;
      else if (code == 32) { // space bar
          cv::Mat crop = frame(box); //crops image to just inside box
          cv::imwrite(std::string("../out") + std::to_string(i++) + ".png", crop); //saves file as "out(integer).png"
      }
  }

  vid_in.release();

  return 0;
}
