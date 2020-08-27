//
// Created by YangYuqi on 2020/7/27.
//

#ifndef LCT2_DETECTOR_H
#define LCT2_DETECTOR_H

#endif //LCT2_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "util.h"
#include <random>

class detector
{
public:

    float ratio;
    cv::Size t_sz;
    int nbin;
    cv::Size target_sz, image_sz;
    cv::Ptr<cv::ml::SVM> det;
    cv::Mat w, svidx;
    double b;
    float angles[4] = {0.15, 0.1, -0.1, -0.15};

    int all_frame_window_num = 60;
    std::default_random_engine e;

    //intialize the detector
    void init(cv::Size target_sz ,cv::Size image_sz);

    //get the needed featuure
    cv::Mat get_feature(cv::Mat image_o);

    //get the needed feature and label
    std::vector<cv::Mat> get_sample(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, float scaleFactor, int step = 1);

    //train the detector
    void train(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, bool online);

    //training with high quality samples
    void hq_train(std::vector<cv::Mat> &hightquality, std::vector<int> &labels);

    //all frame search
    std::pair<std::vector<cv::Mat>, std::vector<cv::Point> > all_frame_search(const cv::Mat &image, float scaleFactor);
};