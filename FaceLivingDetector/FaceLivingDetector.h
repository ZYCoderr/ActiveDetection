#pragma once
#include <string>
#include <opencv2/opencv.hpp>
namespace FaceLivingDetector {

	
	enum Action {		
		shake, 
		open_mouth,
		blink
	};

	/*
		初始化监测器
		model_path: 模型地址
		返回值:		0表示成功，-1表示失败
	*/
	int initDetector(const std::string& detector_model_path, const std::string& resnet_model_path);

	/*
		开始监测
		frame:		视频流
		action:		希望监测的动作
		timeout:	超时时间
		begin:		是否是第一帧
		返回值:		-1表示监测失败，0表示正在检测，1表示检测成功，
	*/
	int aliveDetect(cv::Mat& frame, Action action, double timeout, bool begin);

	cv::Rect findFaceRect(const cv::Mat& frame);

	double faceCompare(const cv::Mat& face1, const cv::Mat& face);


};

