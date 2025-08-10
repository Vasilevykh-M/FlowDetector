#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "include/YOLOv11ONNXDetector.h"

YOLOv11ONNXDetector::YOLOv11ONNXDetector(
    const std::string& modelPath,
    int inputWidth,
    int inputHeight,
    float confThreshold,
    float nmsThreshold,
    std::vector<int> aviable_classes
) : inpWidth(inputWidth), inpHeight(inputHeight),
    confThreshold(confThreshold), nmsThreshold(nmsThreshold), aviable_classes(aviable_classes)
{
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

Detection YOLOv11ONNXDetector::detect(cv::Mat& image) {
    cv::Mat blob;
    preprocess(image, blob);
    std::vector<cv::Mat> outputs;
    predict(blob, outputs);

    return postprocess(image, outputs);
}

void YOLOv11ONNXDetector::preprocess(cv::Mat& image, cv::Mat& blob)
{
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(), true, false);
}

void YOLOv11ONNXDetector::predict(cv::Mat& image, std::vector<cv::Mat>& outputs)
{
    net.setInput(image);
    net.forward(outputs, net.getUnconnectedOutLayersNames());
}

Detection YOLOv11ONNXDetector::postprocess(cv::Mat& image, std::vector<cv::Mat>& outputs) {
    cv::Mat out = outputs[0];

    if (out.dims == 3) {
        if (out.size[1] < out.size[2]) {
            out = out.reshape(1, { out.size[1], out.size[2] }).t();
        }
        else {
            out = out.reshape(1, out.size[1]);
        }
    }

    int num_preds = out.rows;
    int num_attrs = out.cols;
    int num_classes = num_attrs - 4;

    float x_factor = static_cast<float>(image.cols) / inpWidth;
    float y_factor = static_cast<float>(image.rows) / inpHeight;
    float bestConf = 0.0f;
    int left = 0, top = 0, width = 0, height = 0;
    int class_id = 2;

    for (int i = 0; i < num_preds; i++) {
        float cx = out.at<float>(i, 0);
        float cy = out.at<float>(i, 1);
        float w = out.at<float>(i, 2);
        float h = out.at<float>(i, 3);

        cv::Mat scores = out.row(i).colRange(4, num_attrs);
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

        if (std::find(aviable_classes.begin(), aviable_classes.end(), classIdPoint.x) == aviable_classes.end())
            continue;

        if (maxClassScore < confThreshold)
            continue;

        if (bestConf > maxClassScore)
            continue;

        bestConf = maxClassScore;

        left = static_cast<int>((cx - w / 2) * x_factor);
        top = static_cast<int>((cy - h / 2) * y_factor);
        width = static_cast<int>(w * x_factor);
        height = static_cast<int>(h * y_factor);

        left = std::max(0, left);
        top = std::max(0, top);
        width = std::min(width, image.cols - left);
        height = std::min(height, image.rows - top);
        class_id = static_cast<float>(maxClassScore);
    }
    Detection detections = Detection({ class_id, bestConf, cv::Rect(left, top, width, height) });
    return detections;
}
