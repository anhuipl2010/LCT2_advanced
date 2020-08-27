//
// Created by YangYuqi on 2020/7/22.
//

#pragma once

#ifndef LCT2_LCT2TRACKER_H
#define LCT2_LCT2TRACKER_H

#endif //LCT2_LCT2TRACKER_H

#include <opencv2/opencv.hpp>
#include <string>

#include "util.h"
#include"fhog.h"
#include "assignToBins1.h"
#include "detector.h"

const int featuremax= 300;

cv::Mat doWork(cv::InputArray _src,cv::Size ksize,int nbins);

//complex calculating
cv::Mat multichafft(const cv::Mat &input, bool inverse);
cv::Mat complexDivision(const cv::Mat& a, const cv::Mat &b);
cv::Mat complexDivisionReal(const cv::Mat &a, const cv::Mat& b);
cv::Mat complexMultiplication(const cv::Mat& a, const cv::Mat &b, bool conj);

class LCT2tracker
{
public:
    //Constructor
    LCT2tracker();

    //initialize the tracker
    void init(const cv::Rect &roi, cv::Mat Image);

    //get the feature
    cv::Mat get_feature(const cv::Mat &Image, bool hann);

    //initialize the size of window
    void search_window(int target_x, int target_y, int image_x, int image_y);

    //create hanning map
    cv::Mat create_hanning();

    //create a gaussian Peak
    cv::Mat create_gaussian_label(float sigma, int col, int row);

    //do the gaussian correlation, get the kernelized function
    cv::Mat gaussian_correlation(const cv::Mat &xf, const cv::Mat &yf, float sigma);

    //get the scaled sample
    cv::Mat get_scale_sample(const cv::Mat &image, cv::Rect base_target,const float *scale_factor, cv::Size model_sz, bool window);

    std::pair<cv::Point, std::pair<float, float>> do_correlation(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, bool window, bool app);

    //online trainning and detection
    cv::Point detect(cv::Mat image);

    //use detector to find pos
    std::pair<cv::Point, std::pair<float, float> >  refine_pos(cv::Mat image, int pos_x, int pos_y, bool app);

    //get the high quality samples for the svm
    void get_high_quality(const cv::Mat &image, int pos_x, int pos_y, cv::Size window_sz);

    //all frame search when lost the object
    std::pair<cv::Point, float> search_from_whole(const cv::Mat &image);

    bool isgray = true;

    float padding; //area surrounding the target
    float lambda; //regularization
    float output_sigma_factor; //spatial bandwidth
    float interp_factor;
    float scale_inter;
    float kernal_sigma;
    bool resize_image;
    int window_x;int window_y;//size of the motion window
    int app_x;int app_y;//size of the appreance window
    float output_sigma;
    float alpha_pt;

    //hog features
    int hog_orientations;
    int cell_size; //hog grid cell
    int window_size; //hoi local region
    int nbins; //bins of HOI

    //threshold
    double motion_thresh;
    double appearance_thresh_min;
    double appearance_thresh_max;
    double accept_thresh;
    double max_mean, min_mean;

    //threshold for APCE
    double APCE_motion_thresh;
    double APCE_appearance_thresh_max;
    double APCE_appearance_thresh_min;
    double APCE_accept_thresh;
    double APCE_max_mean, APCE_min_mean;

    cv::Rect _roi;

    //scalefactor
    int nScale;
    float scale_factor[40];
    float scale_sigma;
    cv::Size scale_model_sz;
    float currentscalefactor;
    float max_scalefactor;
    float min_scalefactor;
    float scale_step;

    //detector
    int lost = 0;
    detector det;
    float m_response;
    std::vector<cv::Mat> highquality;
    std::vector<int> labels;
    float det_scalefactor[10];
    float det_scalestep;
    int det_nScale;

    int count = 0;
    std::pair<double, double> responce_sum = std::make_pair(0, 0);
    std::pair<double, double> responce_mean = std::make_pair(0, 0);
private:
    int size_patch[3];

    //feature data
    cv::Mat win_xf;
    cv::Mat _alphaf;
    cv::Mat app_xf;
    cv::Mat app_alphaf;
    cv::Mat sf_num;
    cv::Mat sf_den;

    //gaussian target
    cv::Mat tar;
    cv::Mat app_tar;
    cv::Mat scale_tar;
};