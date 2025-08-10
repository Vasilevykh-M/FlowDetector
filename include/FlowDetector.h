#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "YOLOv11ONNXDetector.h"

class FlowDetector
{
public:
	FlowDetector(const std::string& modelPath, cv::Size win_size, int grid_size);
	~FlowDetector();
	void predict_flow(std::vector<cv::String> filenames, std::string output_path);

private:
	std::vector<cv::Point2f> points_1;
	std::vector<cv::Point2f> points_2;
	std::vector<uchar> status;
	cv::Mat img;
	cv::Mat gray_1;
	cv::Mat gray_2;
	YOLOv11ONNXDetector* bbox_detector;
	cv::Size win_size;
	int grid_size;

	void get_grid_bbox(cv::Rect box);

	void get_optical_flow();

	void draw_flow(cv::Rect bbox, std::string outputh_path);
};