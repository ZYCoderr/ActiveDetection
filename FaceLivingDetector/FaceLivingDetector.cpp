#include "stdafx.h"
#include "FaceLivingDetector.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <time.h>
#include <type_traits>
namespace FaceLivingDetector {
	
	using namespace dlib;

	template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
	using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

	template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
	using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

	template <int N, template <typename> class BN, int stride, typename SUBNET>
	using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

	template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
	template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

	template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
	template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
	template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
	template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
	template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

	using anet_type = loss_metric<fc_no_bias<128, dlib::avg_pool_everything<
		alevel0<
		alevel1<
		alevel2<
		alevel3<
		alevel4<
		max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
		input_rgb_image_sized<150>
		>>>>>>>>>>>>;

	enum HeadState {
		normal = 1,
		turn_left = 2,
		turn_right = 4
	};

	enum MouseState {
		mouse_close = 1,
		mouse_open = 2
	};

	enum EyesState {
		eyes_open = 1,
		eyes_close = 2
	};

	dlib::frontal_face_detector g_detector; // 人脸探测器
	dlib::shape_predictor g_predictor; // 特征点计算器
	anet_type g_net; // 用于计算脸部特征向量的神经网络

	bool g_is_inited = false;
	bool g_check_successful = false;

	clock_t g_time_start, g_time_cur;

	int g_ok_cnt = 0; // 检测到的状态与期望的状态相符，成功次数会增加
	int g_except_state = 0; 
	
	const double eyes_ar_thresh = 0.43; // 如果AR大于它，则认为眼睛是睁开的。如果AR值小于它，则认为眼睛是闭上的。 

	int initDetector(const std::string& detector_model_path, const std::string& resnet_model_path)
	{
		if (g_is_inited) return 0;
		if (detector_model_path.empty() || resnet_model_path.empty()) return -1;

		g_detector = dlib::get_frontal_face_detector();
		dlib::deserialize(detector_model_path.c_str()) >> g_predictor;
		dlib::deserialize(resnet_model_path.c_str()) >> g_net;
		g_is_inited = true;
		return 0;
	}

	double calcDistance(const cv::Point& pt1, const cv::Point& pt2)
	{
		return sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
	}

	double calcDistance(const dlib::full_object_detection& shape, int idx_from, int idx_to)
	{
		cv::Point pt1 = cv::Point(shape.part(idx_from).x(), shape.part(idx_from).y());
		cv::Point pt2 = cv::Point(shape.part(idx_to).x(), shape.part(idx_to).y());
		return calcDistance(pt1, pt2);	
	}

	/*
		下标从0开始
		4号点：	嘴部右侧脸颊
		48号点：嘴唇右侧
		12号点：嘴部左侧脸颊		
		54号点：嘴唇左侧
	*/
	HeadState headCheck(const dlib::full_object_detection& shape)
	{
		double dist_left = calcDistance(shape, 4, 48);
		double dist_right = calcDistance(shape, 12, 54);
		double scale = dist_left / dist_right;

		if (scale <= 0.3) return HeadState::turn_left;
		else if (scale >= 2.5) return HeadState::turn_right;
		else return HeadState::normal;
	}

	/*
		下标从0开始
		33号点：鼻子底部
		62号点：上嘴唇底部
		66号点：上嘴唇顶部
	*/
	MouseState mouthCheck(const dlib::full_object_detection& shape)
	{
		double dist_top = calcDistance(shape, 33, 62);
		double dist_bot = calcDistance(shape, 33, 66);
		double scale = dist_top / dist_bot;

		if (scale >= 0.9 && scale <= 1.1) return MouseState::mouse_close;
		return MouseState::mouse_open;
	}

	double calcARValue(const std::vector<cv::Point>& pts)
	{
		return (calcDistance(pts[1], pts[5]) + calcDistance(pts[2], pts[4])) / (2 * calcDistance(pts[0], pts[3]));
	}

	// AR阈值法
	// 参考资料: https://blog.csdn.net/hongbin_xu/article/details/79033116
	EyesState eyesCheck(const dlib::full_object_detection& shape)
	{
		std::vector<cv::Point> eye_pts_left;		
		for (int i = 36; i < 42; i++) {
			eye_pts_left.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
		double ar_left = calcARValue(eye_pts_left);

		std::vector<cv::Point> eye_pts_right;
		for (int i = 42; i < 48; i++) {
			eye_pts_right.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
		}
		double ar_right = calcARValue(eye_pts_right);

		if (ar_left + ar_right > eyes_ar_thresh) return EyesState::eyes_open;
		return EyesState::eyes_close;
	}

	int aliveDetect(cv::Mat& frame, Action action, double timeout, bool begin)
	{
		if (!g_is_inited) return -1;

		if (begin) g_time_start = clock();
		else {
			g_time_cur = clock();
			if (g_time_cur - g_time_start > timeout) {
				return g_check_successful ? 1 : -1;
			}
		}

		dlib::cv_image<dlib::bgr_pixel> img(frame);
		std::vector<dlib::rectangle> faces = g_detector(img);

		if (!faces.empty()) {
			// 获取脸部68个特征点
			// 特征点分布图: https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
			dlib::full_object_detection shape = g_predictor(img, faces[0]);
			int ret = 0;
			switch (action) {

				case FaceLivingDetector::shake:
					if (begin) {
						g_ok_cnt = 0;
						g_check_successful = false;
						g_except_state = HeadState::turn_left | HeadState::turn_right;
					} 
					ret = headCheck(shape);
					if (ret & g_except_state) {
						g_ok_cnt++;
						if (g_ok_cnt > 3) g_check_successful = true;
						if (ret == HeadState::turn_left) g_except_state = HeadState::turn_right;
						else if (ret == HeadState::turn_right) g_except_state = HeadState::turn_left;
					}
					break;

				case FaceLivingDetector::open_mouth:
					if (begin) {
						g_ok_cnt = 0;
						g_check_successful = false;
						g_except_state = MouseState::mouse_close;
					}
					ret = mouthCheck(shape);
					if (ret & g_except_state) {
						g_ok_cnt++;
						if (g_ok_cnt > 1) g_check_successful = true;
						if (ret == MouseState::mouse_close) g_except_state = MouseState::mouse_open;
						else if (ret == MouseState::mouse_open) g_except_state = MouseState::mouse_close;
					}
					break;

				case FaceLivingDetector::blink:
					if (begin) {
						g_ok_cnt = 0;
						g_check_successful = false;
						g_except_state = EyesState::eyes_open;
					}
					ret = eyesCheck(shape);
					if (ret & g_except_state) {
						g_ok_cnt++;
						if (g_ok_cnt > 3) g_check_successful = true;
						if (ret == EyesState::eyes_open) g_except_state = EyesState::eyes_close;
						else if (ret == EyesState::eyes_close) g_except_state = EyesState::eyes_open;
					}
					break;
			}
		}
		if (!g_check_successful) return 0; 
		return 1;
	}

	cv::Rect findFaceRect(const cv::Mat & frame)
	{
		dlib::cv_image<dlib::bgr_pixel> img(frame);
		std::vector<dlib::rectangle> faces_rect = g_detector(img);
		if (faces_rect.empty()) return cv::Rect();

		dlib::point tl = faces_rect[0].tl_corner();
		dlib::point br = faces_rect[0].br_corner();		
		return cv::Rect(cv::Point(tl.x(), tl.y()), cv::Point(br.x(), br.y())); 
	}

	int getFaceDescriptiot(const cv::Mat& face, dlib::matrix<float, 0, 1>& descriptor)
	{
		dlib::cv_image<dlib::bgr_pixel> img(face);
		std::vector<dlib::rectangle> faces_rect = g_detector(img);
		if (faces_rect.empty()) return -1;

		dlib::full_object_detection shape = g_predictor(img, faces_rect[0]);
		
		std::vector<matrix<rgb_pixel>> faces;
		matrix<rgb_pixel> face_chip;
		extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(std::move(face_chip));

		descriptor = g_net(faces)[0];
		return 0;
	}

	double faceCompare(const cv::Mat& face1, const cv::Mat & face2)
	{
		int ret = 0;

		dlib::matrix<float, 0, 1> face1_descriptor;
		ret = getFaceDescriptiot(face1, face1_descriptor);
		if (ret < 0) return 0;

		dlib::matrix<float, 0, 1> face2_descriptor;
		ret = getFaceDescriptiot(face2, face2_descriptor);
		if (ret < 0) return 0;

		return 1 - dlib::length(face1_descriptor - face2_descriptor);
	}

};