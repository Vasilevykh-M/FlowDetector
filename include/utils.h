#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

std::vector<cv::String> get_files(std::string input_dir)
{
    std::vector<cv::String> filenames;
    cv::glob(input_dir + "/*.png", filenames, false);
    if (filenames.empty()) {
        cv::glob(input_dir + "/*.jpg", filenames, false);
    }
    if (filenames.empty()) {
        std::cerr << "No image files found in " << input_dir << std::endl;
        return filenames;
    }

    sort(filenames.begin(), filenames.end());

    return filenames;
}