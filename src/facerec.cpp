#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utils/logger.hpp" //new
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <future>

// Global variable to determine if mouse is dragging
bool dragging = false;

// Region of interest on the frame, this is where the facial recgonition takes place. Global scope so it can be used by the DragRect function
cv::Rect roi(228, 128, 184, 224);

// Function for checking if the ROI is still within the boundaries of the frame. Crated this to try and keep the code as readable as possible.
void CheckBoundaries(cv::Mat& frame)
{
  if (roi.x < 0)
  {
    roi.x = 0;
  }
  
  if (roi.y < 0)
  {
    roi.y = 0;
   
  }
  
  if (roi.x + roi.width > frame.cols)
  {
      roi.x = frame.cols - roi.width;
  }
  
  if (roi.y + roi.height > frame.rows)
  {
      roi.y = frame.rows - roi.height;
  }
}

void DragRect(int event, int x, int y, int flags, void* userdata)
{

  // Static cast the userdata to a cv::Mat* so we can check the bounds of the frame
  cv::Mat* frame = static_cast<cv::Mat*>(userdata);
  
   // Used to detremine wheter or not th euser is moving the mous whilst holding left button down on rectangle
    if (event == cv::EVENT_LBUTTONDOWN)
    {
      if(roi.contains(cv::Point(x, y)))
      {
	dragging = true;
      }
      
    }else if (event == cv::EVENT_MOUSEMOVE && dragging)
    {
      roi.x = x - roi.width /2;
      roi.y = y - roi.height /2;

      // Ensure the ROI stays within the frame boundaries
      CheckBoundaries(*frame);
        
    }else if (event == cv::EVENT_LBUTTONUP)
    {
      dragging = false;
    }
}

// Function for prediciting face within the ROI, this will return the label of whatever face it predicts.
int Predict(cv::Ptr<cv::face::BasicFaceRecognizer>& model, cv::Mat& frame)
{

  cv::Mat crop = frame(roi); //crops image to just inside box
  cv::resize(crop, crop, cv::Size(92, 112), cv::INTER_LINEAR);
  cv::cvtColor(crop, crop, cv::COLOR_BGR2GRAY);

  cv::imwrite("out.pgm", crop);

  int predictedLabel = model->predict(crop);

  return predictedLabel;
	  
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
      vid_in.release();
      return -1;
  }

  cv::namedWindow(win_name);

  cv::setMouseCallback(win_name, DragRect, &frame);

  while (true)
  {

    vid_in >> frame;

      // Async Function to predict whose face is within the ROI
      std::future<int> predictedLabel = std::async(std::launch::async, &Predict, std::ref(model), std::ref(frame));
      // Text to display under the rectangle based on which face has been detected by the model
      std::string label;
      
      // Check the returned label value to determine if the face detected is a member of this group
      int value = predictedLabel.get();
      switch(value)
      {
      case 41:
	label = "Oscar";
	break;
      case 42:
	label = "Matthew";
	break;
      case 43:
	label = "Greer";
	break;
      case 44:
	label = "Lucas";
	break;
      default:
	label = std::to_string(value) + " - Not a Team Member";
	break;
      }

      // Display the label on screen just under the rectangle
      cv::putText(frame, label, cv::Point(roi.x, roi.y + roi.height + 20),
		  cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0), 2);
     
      
      /* Draw border aroud the rect object, this is done so the users can see what they are moving, highlight the
      rectangle a different colour when dragging os occuring, just to let the user know they are dragging it,
      only apply the filter when not dragging the rectangle.
      */
      if(dragging)
      {
	cv::rectangle(frame, roi, cv::Scalar(255, 0, 255), 2);
      }else
      {
	cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);

	// Creates GaussianBlur within the ROI rectangle
	cv::GaussianBlur(frame(roi), frame(roi), cv::Size(51, 51), 0);
      }
      
      imshow(win_name, frame);

      
      int code = cv::waitKey(1000 / fps); // how long to wait for a key (msecs)
      if (code == 27) // escape. See http://www.asciitable.com/
          break;
 
      
	
      
  }
  vid_in.release();

  return 0;
}

