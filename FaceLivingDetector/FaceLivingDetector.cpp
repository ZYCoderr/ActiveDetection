#include "stdafx.h"
#include "FaceLivingDetector.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <time.h>

namespace FaceLivingDetector {
	
	enum HeadState {
		normal = 1,
		turn_left = 2,
		turn_right = 4
	};

	enum MouseState {
		mouse_colse = 1,
		mouse_open = 2
	};

	enum EyesState {
		eyes_open = 1,
		eyes_colse = 2
	};

	dlib::frontal_face_detector detector;
	dlib::shape_predictor predictor;

	bool g_is_inited = false;
	bool check_successful = false;

	clock_t g_time_start, g_time_cur;

	int g_ok_cnt = 0; // 检测到的状态与期望的状态相符，成功次数会增加
	int g_except_state = 0; 
	
	const double eyes_ar_thresh = 0.43; // 如果AR大于它，则认为眼睛是睁开的。如果AR值小于它，则认为眼睛是闭上的。 

	int init(const std::string& model_path)
	{
		if (g_is_inited) return 0;
		if (model_path.empty()) return -1;
		
		detector = dlib::get_frontal_face_detector();
		dlib::deserialize(model_path.c_str()) >> predictor;
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

		if (scale >= 0.9 && scale <= 1.1) return MouseState::mouse_colse;
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
		return EyesState::eyes_colse;
	}

	int execute(cv::Mat& frame, Action action, double timeout, bool begin)
	{
		if (!g_is_inited) return -1;

		if (begin) g_time_start = clock();
		else {
			g_time_cur = clock();
			if (g_time_cur - g_time_start > timeout) {
				return check_successful ? 1 : -1;
			}
		}

		dlib::cv_image<dlib::bgr_pixel> img(frame);
		std::vector<dlib::rectangle> faces = detector(img);
		std::stringstream ss;

		if (!faces.empty()) {
			// 获取脸部68个特征点
			// 特征点分布图: https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
			dlib::full_object_detection shape = predictor(img, faces[0]);
			int ret = 0;
			switch (action) {

				case FaceLivingDetector::shake:
					if (begin) {
						g_ok_cnt = 0;
						check_successful = false;
						g_except_state = HeadState::turn_left | HeadState::turn_right;
					} 
					ret = headCheck(shape);
					if (ret & g_except_state) {
						g_ok_cnt++;
						if (g_ok_cnt > 3) check_successful = true;
						if (ret == HeadState::turn_left) g_except_state = HeadState::turn_right;
						else if (ret == HeadState::turn_right) g_except_state = HeadState::turn_left;
					}
					break;

				case FaceLivingDetector::open_mouth:
					if (begin) {
						g_ok_cnt = 0;
						check_successful = false;
						g_except_state = MouseState::mouse_colse;
					}
					ret = mouthCheck(shape);
					if (ret & g_except_state) {
						g_ok_cnt++;
						if (g_ok_cnt > 1) check_successful = true;
						if (ret == MouseState::mouse_colse) g_except_state = MouseState::mouse_open;
						else if (ret == MouseState::mouse_open) g_except_state = MouseState::mouse_colse;
					}
					break;

				case FaceLivingDetector::blink:
					if (begin) {
						g_ok_cnt = 0;
						check_successful = false;
						g_except_state = EyesState::eyes_open;
					}
					ret = eyesCheck(shape);
					if (ret & g_except_state) {
						g_ok_cnt++;
						if (g_ok_cnt > 3) check_successful = true;
						if (ret == EyesState::eyes_open) g_except_state = EyesState::eyes_colse;
						else if (ret == EyesState::eyes_colse) g_except_state = EyesState::eyes_open;
					}
					break;
			}
		}
		if (!check_successful) return 0; 
		return 1;
	}
};