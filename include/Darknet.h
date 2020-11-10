#ifndef DARKNET_H
#define DARKNET_H
/*******************************************************************************
* 
* Author : walktree
* Email  : walktree@gmail.com
*
* A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. 
* It's fast, easy to be integrated to your production, and supports CPU and GPU computation. Enjoy ~
*
*******************************************************************************/

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

//Image
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
#define YOLO_TENSOR_W 416
#define YOLO_TENSOR_H 416

#define B_BOX_ENLARGE_SCALE 0.2f

struct Darknet : torch::nn::Module {

public:

	Darknet(const char *conf_file, torch::Device *device);

	map<string, string>* get_net_info();

	void load_weights(const char *weight_file);

	torch::Tensor forward(torch::Tensor x);

	/**
	 *  对预测数据进行筛选
	 */
	torch::Tensor write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf = 0.4);

	std::vector<cv::Rect> postprocess(torch::Tensor output, cv::Mat frame, double orig_w, double orig_h, double resize_ratio);

private:

	torch::Device *_device;

	vector<map<string, string>> blocks;

	torch::nn::Sequential features;

	vector<torch::nn::Sequential> module_list;

    // load YOLOv3 
    void load_cfg(const char *cfg_file);

    void create_modules();

    int get_int_from_cfg(map<string, string> block, string key, int default_value);

    string get_string_from_cfg(map<string, string> block, string key, string default_value);


};
#endif