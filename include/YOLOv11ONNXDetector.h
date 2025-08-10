#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YOLOv11ONNXDetector {
public:
    YOLOv11ONNXDetector(
        const std::string& modelPath,
        int inputWidth = 640,
        int inputHeight = 640,
        float confThreshold = 0.2,
        float nmsThreshold = 0.8,
        std::vector<int> aviable_classes = std::vector<int>({ 0, 1, 2, 3, 4, 5, 6, 7, 8 })
    );

    Detection detect(cv::Mat& image);

private:
    int inpWidth, inpHeight;
    float confThreshold;
    float nmsThreshold;
    cv::dnn::Net net;
    std::vector<int> aviable_classes;

    Detection postprocess(cv::Mat& image, std::vector<cv::Mat>& outputs);
    void preprocess(cv::Mat& image, cv::Mat& blob);
    void predict(cv::Mat& image, std::vector<cv::Mat>& outputs);
};
