#include <opencv2/opencv.hpp>
#include "FlowDetector.h"
#include "utils.h"

int main(int argc, char* argv[]) 
{
    if(argc != 4)
        return -1;

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    std::vector <cv::String > filenames = get_files(input_dir);
    FlowDetector flow_detector = FlowDetector(argv[3], cv::Size(80, 80), 25);
    if (filenames.empty()) {
        std::cerr << "No image files found in " << input_dir << std::endl;
        return -1;
    }
    flow_detector.predict_flow(filenames, output_dir);
    return 0;
}