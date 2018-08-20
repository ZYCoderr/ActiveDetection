#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/dnn.h>
namespace FaceLivingDetector {

	
	enum Action {		
		shake, 
		open_mouth,
		blink
	};

	/*
		��ʼ�������
		model_path: ģ�͵�ַ
		����ֵ:		0��ʾ�ɹ���-1��ʾʧ��
	*/
	int initDetector(const std::string& detector_model_path, const std::string& resnet_model_path);

	/*
		��ʼ���
		frame:		��Ƶ��
		action:		ϣ�����Ķ���
		timeout:	��ʱʱ��
		begin:		�Ƿ��ǵ�һ֡
		����ֵ:		-1��ʾ���ʧ�ܣ�0��ʾ���ڼ�⣬1��ʾ���ɹ���
	*/
	int aliveDetect(cv::Mat& frame, Action action, double timeout, bool begin);

	cv::Rect findFaceRect(const cv::Mat& frame);

	double faceCompare(const cv::Mat& face1, const cv::Mat& face);


};

