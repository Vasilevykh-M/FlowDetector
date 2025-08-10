#include <opencv2/opencv.hpp>
#include "include/FlowDetector.h"
#include "include/utils.h"

int main() 
{
    std::string input_dir = "C:\\Users\\Михаил\\Desktop\\Магистратура\\test\\data_of\\data_of";
    std::string output_dir = "C:\\Users\\Михаил\\Desktop\\Магистратура\\test\\result";

    std::vector <cv::String > filenames = get_files(input_dir);

    FlowDetector flow_detector = FlowDetector("C:\\Users\\Михаил\\Downloads\\yolo11n.onnx", cv::Size(80, 80), 25);

    if (filenames.empty()) {
        std::cerr << "No image files found in " << input_dir << std::endl;
        return -1;
    }
    flow_detector.predict_flow(filenames, output_dir);
    return 0;
}