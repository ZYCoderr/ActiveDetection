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
		��ʼ�������
		model_path: ģ�͵�ַ
		����ֵ:		0��ʾ�ɹ���-1��ʾʧ��
	*/
	int init(const std::string& model_path);

	/*
		��ʼ���
		frame:		��Ƶ��
		action:		ϣ�����Ķ���
		timeout:	��ʱʱ��
		begin:		�Ƿ��ǵ�һ֡
		����ֵ:		-1��ʾ���ʧ�ܣ�0��ʾ���ڼ�⣬1��ʾ���ɹ���
	*/
	int execute(cv::Mat& frame, Action action, double timeout, bool begin);
};

