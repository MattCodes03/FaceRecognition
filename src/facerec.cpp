#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utils/logger.hpp" //new
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
    else if (event == cv::EVENT_MBUTTONDOWN)
    {
        std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;

    }
    
}

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

  //new - this has been taken from 04_capture_show_video.cpp from the open cv lab
  cv::Mat frame;
  cv::Mat grey_scale;
  cv::Mat black_white;
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
  cv::Point topLeft = cv::Point(228, 128);
  cv::Point bottomRight = cv::Point(412, 352);
  cv::Mat temp;
  cv::Mat crop;

  while (1) {
      
      vid_in >> frame;
      vid_in >> temp; //temp is used to have unfiltered view

      cv::Rect roi = cv::Rect(topLeft, bottomRight); //creates rectangle object
      cv::GaussianBlur(frame(roi), frame(roi), cv::Size(51, 51), 0); //creates Gaussian Blur only inside box

      cv::Point cursorPos;
      
      cv::setMouseCallback(win_name, CallBackFunc, NULL);

      imshow(win_name, frame);
      
      int code = cv::waitKey(1000 / fps); // how long to wait for a key (msecs)
      if (code == 27) // escape. See http://www.asciitable.com/
          break;
      else if (code == 32) { // space bar
          cv::Mat crop = temp(roi); //crops image to just inside box
          cv::resize(crop, crop, cv::Size(92, 112), cv::INTER_LINEAR); //sca
          cv::cvtColor(crop, crop, cv::COLOR_BGR2GRAY);
          cv::imwrite(std::string("../out")+ ".pgm", crop); //saves file as "out.png"
          cv::Mat testSample = cv::imread(std::string("../out") + ".pgm", cv::IMREAD_GRAYSCALE); //reads output image
          int predictedLabel = model->predict(testSample); //predicts label of face
          std::cout << "\nPredicted class = " << predictedLabel << '\n';
      }
  }

  vid_in.release();

  return 0;
}
