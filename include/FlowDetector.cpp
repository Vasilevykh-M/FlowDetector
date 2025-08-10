#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include "include/FlowDetector.h"

FlowDetector::FlowDetector(const std::string& modelPath, cv::Size win_size, int grid_size) : win_size(win_size), grid_size(grid_size)
{
    bbox_detector = new YOLOv11ONNXDetector(modelPath);
}

FlowDetector::~FlowDetector()
{
    delete bbox_detector;
    points_1.~vector();
    points_2.~vector();
    status.~vector();
    img.deallocate();
    gray_1.deallocate();
    gray_2.deallocate();
}

void FlowDetector::predict_flow(std::vector<cv::String> filenames, std::string output_path)
{
    img = cv::imread(filenames[0]);
    cvtColor(img, gray_1, cv::COLOR_BGR2GRAY);
    cv::Mat img2;

    for (size_t i = 0; i < filenames.size() - 1; i++) {
        img2 = cv::imread(filenames[i + 1]);
        Detection det = bbox_detector->detect(img);
        cvtColor(img2, gray_2, cv::COLOR_BGR2GRAY);
        get_grid_bbox(det.box);
        get_optical_flow();
        std::stringstream ss;
        ss << output_path << "/flow_" << std::setfill('0') << std::setw(4) << i << ".png";
        draw_flow(det.box, ss.str());

        gray_1 = gray_2.clone();
        img = img2.clone();
        points_1.clear();
        points_2.clear();
        status.clear();
    }
}

void FlowDetector::get_grid_bbox(cv::Rect box)
{
    cv::Size img_size = img.size();
    int start_y = box.height == 0 ? 0 : box.y;
    int start_x = box.width == 0 ? 0 : box.x;
    int h = box.height == 0 ? img_size.height : box.y + box.height;
    int w = box.width == 0 ? img_size.width : box.x + box.width;
    for (int y = start_y + grid_size; y < h; y += grid_size) {
        for (int x = start_x + grid_size; x < w; x += grid_size) {
            points_1.push_back(cv::Point2f(x, y));
        }
    }
}

void FlowDetector::get_optical_flow(
)
{
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        gray_1, gray_2,
        points_1, points_2,
        status, err,
        win_size, 4,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
        0, 1e-4
    );
}

void FlowDetector::draw_flow(
    cv::Rect bbox,
    std::string outputh_path
)
{
    for (size_t j = 0; j < points_1.size(); j++) {
        if (status[j]) {
            arrowedLine(img, points_1[j], points_2[j], cv::Scalar(0, 255, 0), 2, 8, 0, 0.2);
        }
    }
    cv::rectangle(img, bbox, cv::Scalar(0, 255, 255), 2);
    imwrite(outputh_path, img);
}