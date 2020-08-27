//
// Created by YangYuqi on 2020/7/27.
//

#include "detector.h"

double round(double r)
{
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

void rotate(const cv::Mat &srcImage, cv::Mat &destImage, double angle)
{
    cv::Point2f center(srcImage.cols / 2, srcImage.rows / 2);
    cv::Mat M = (cv::Mat_<float>(2, 3)<<1, std::tan(angle), 0, 0, 1, 0);
    cv::warpAffine(srcImage, destImage, M, cv::Size(srcImage.cols, srcImage.rows));
}


void detector::init(cv::Size target_sz, cv::Size image_sz) {
    e.seed(1);
    float target_max_win = 256;
    ratio = std::sqrt(target_max_win/target_sz.area());

    t_sz.width = round(target_sz.width*ratio);
    t_sz.height = round(target_sz.height*ratio);

    nbin = 32;

    this->target_sz = target_sz;
    this->image_sz = image_sz;


    det = cv::ml::SVM::create();
}

cv::Mat detector::get_feature(cv::Mat image_o) {
//    int nth;
//    cv::Mat image = image_o.clone(), f;
//    if(image.channels() == 3)
//    {
//        cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
//        nth = 4;
//        std::vector<cv::Mat> Lab;
//        cv::split(image, Lab);
//        f = Lab[0].clone();
//    }
//    else {
//        nth = 8;
//        f = image.clone();
//    }
//    int ksize = 4;
//    cv::Mat f_iif = 255 - doWork(f, cv::Size(ksize, ksize), nbin);
//    std::vector<cv::Mat> ans, tmp;
//    cv::split(image, tmp);
//    for(int i = 1; i <= nth; i++)
//    {
//        float thr = i/float(nth + 1)*255;
//        ans.push_back(f_iif >= thr);
//    }
//    for(int k = 0; k < tmp.size(); k++)
//    {
//        for(int i = 1; i <= nth; i++)
//        {
//            float thr = i/float(nth + 1)*255;
//            ans.push_back(tmp[k] >= thr);
//        }
//    }
//    cv::Mat out;
//    cv::merge(ans, out);
    return image_o;
}

std::vector<cv::Mat> detector::get_sample(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, float scaleFactor, int step) {
    cv::Mat w_area = get_subwindow(image, pos_x, pos_y, floor(window_sz.height), floor(window_sz.width));
    //cv::imshow("in", w_area);
    //cv::waitKey();
    cv::Mat feat = get_feature(w_area);
    cv::resize(feat, feat, cv::Size(ceil(feat.cols*ratio/scaleFactor), ceil(feat.rows*ratio/scaleFactor)), 0, 0, cv::INTER_NEAREST);
    step = std::min(step, std::min(feat.rows - t_sz.height, feat.cols - t_sz.width) - 1);

    std::vector<cv::Mat> alfeat;
    cv::Mat label(ceil((feat.rows - t_sz.height)/(step + 0.f)), ceil((feat.cols - t_sz.width)/(step + 0.f)), CV_32F, cv::Scalar::all(0));
    cv::Mat yy(label.rows, label.cols, CV_32F), xx(label.rows, label.cols, CV_32F, cv::Scalar::all(0));
    cv::Mat weights(label.rows, label.cols, CV_32F, cv::Scalar::all(0));
    for(int i = 0; i < weights.rows; i++)
        for(int j = 0; j < weights.cols; j++)
            weights.at<float>(i, j) = std::exp(-0.5*((i - weights.rows/2)*(i - weights.rows/2) + (j - weights.cols/2)*(j - weights.cols/2))/(25.0));
    //std::cout<<weights<<std::endl;
    cv::Rect target_rect((feat.cols-t_sz.width)/2, (feat.rows - t_sz.height)/2, t_sz.width, t_sz.height);

    int truerow = label.rows;int truecol = label.cols;int truesta_i = 0;int truesta_j = 0;
    for(int i = 0; i < feat.rows - t_sz.height; i += step)
    {
        if((i + t_sz.height/2 - feat.rows/2)/ratio + pos_x < 0)
        {
            truesta_i++;truerow--;
            continue;
        }
        else if((i + t_sz.height/2 -feat.rows/2)/ratio + pos_x >= image_sz.height)
        {
            truerow--;
            continue;
        }
        for(int j = 0; j < feat.cols - t_sz.width; j += step)
        {
            if(j/step < truesta_j || j/step > truesta_j + truecol - 1)
                continue;
            if((j + t_sz.width/2 -feat.cols/2)/ratio + pos_y < 0)
            {
                truesta_j++;truecol--;
                continue;
            }
            else if((j + t_sz.width/2 -feat.cols/2)/ratio + pos_y >= image_sz.width)
            {
                truecol--;
                continue;
            }
            //std::cout<<i<<" "<<j<<t_sz<<std::endl;
            cv::Rect range(j, i, t_sz.width, t_sz.height);
            cv::Mat localfeat = feat(range).clone().reshape(1, 1);
            alfeat.push_back(localfeat);
            //std::cout<<target_rect<<std::endl<<range<<std::endl;
            label.at<float>(i/step, j/step) = ((range & target_rect).area() + 0.0f)/(range.area() + target_rect.area() - (range & target_rect).area());
            xx.at<float>(i/step, j/step) = i;
            yy.at<float>(i/step, j/step) = j;
        }
    }
    xx = (xx + t_sz.height/2 - feat.rows/2)/ratio + pos_x;
    yy = (yy + t_sz.width/2 - feat.cols/2)/ratio + pos_y;
    xx = xx(cv::Range(truesta_i, truesta_i + truerow), cv::Range(truesta_j, truesta_j + truecol)).clone().reshape(1, 1);;
    yy = yy(cv::Range(truesta_i, truesta_i + truerow), cv::Range(truesta_j, truesta_j + truecol)).clone().reshape(1, 1);
    weights = weights(cv::Range(truesta_i, truesta_i + truerow), cv::Range(truesta_j, truesta_j + truecol)).clone();
    cv::Mat feature(alfeat.size(), alfeat[0].cols, CV_32F, cv::Scalar::all(0));
    for(int i = 0; i < alfeat.size(); i++)
    {
        cv::Mat tmp;
        alfeat[i].convertTo(tmp, CV_32F);
        alfeat[i].copyTo(feature.row(i));
    }
    label = label.reshape(1, 1);
    std::vector<cv::Mat> ans;


    ans.push_back(feature.clone());
    ans.push_back(label.clone().t());
    ans.push_back(xx.clone());
    ans.push_back(yy.clone());
    ans.push_back(weights.clone());
    return ans;
}

void detector::train(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, bool online) {
    //std::cout<<"de"<<std::endl;
    std::vector<cv::Mat> features;
    std::vector<int> labels;
    cv::Mat tar = get_subwindow(image, pos_x, pos_y, window_sz.height, window_sz.width), tarfeat;
    tarfeat = get_feature(tar);
    cv::resize(tarfeat, tarfeat, t_sz, 0, 0, cv::INTER_NEAREST);
    features.push_back(tarfeat.reshape(1, 1));
    labels.push_back(1);
    for(int i = 0; i < 4; i++)
    {
        cv::Mat rotar;
        rotate(tar, rotar, angles[i]);
        cv::Mat rotarfeat = get_feature(rotar);
        cv::resize(rotarfeat, rotarfeat, t_sz, 0, 0, cv::INTER_NEAREST);
        features.push_back(rotarfeat.reshape(1, 1));
        labels.push_back(1);
    }
    //std::cout<<samples[1]<<std::endl;
    int iter =20;
    float posi = 0.5,  nega = 0.1;
    std::uniform_int_distribution<unsigned> x_range(0, image.cols - 1);
    std::uniform_int_distribution<unsigned> y_range(0, image.rows -  1);
    while(iter--)
    {
        cv::Rect nega_rect = cv::Rect(cv::Point(0, 0), window_sz), tar_rect = cv::Rect(pos_y - window_sz.width/2, pos_x - window_sz.height/2, window_sz.width, window_sz.height);
        do
        {
            nega_rect.x = x_range(e);
            nega_rect.y = y_range(e);
            std::uniform_real_distribution<float> scale(0.3*std::min((image.cols - nega_rect.x)/target_sz.width, (image.rows - nega_rect.y)/target_sz.height), 0.7 *
            std::min((image.cols - nega_rect.x)/target_sz.width, (image.rows - nega_rect.y)/target_sz.height));
            float s = scale(e);
            nega_rect.width = floor(target_sz.width * s);
            nega_rect.height = floor(target_sz.height * s);
        }
        while((nega_rect.width == 0 && nega_rect.height == 0) || (tar_rect & nega_rect).area() == tar_rect.area() ||(tar_rect & nega_rect).area()/(tar_rect.area() + nega_rect.area() - (tar_rect & nega_rect).area() + 0.f) > nega);
        cv::Mat nega_sam = image(nega_rect).clone();
//        cv::imshow("n", nega_sam);
//        cv::waitKey();
//        cv::destroyWindow("n");
        cv::Mat nega_feat = get_feature(nega_sam);
        cv::resize(nega_feat, nega_feat, t_sz, 0, 0, cv::INTER_NEAREST);
        features.push_back(nega_feat.reshape(1, 1));
        labels.push_back(-1);
    }
    cv::Mat feat(features.size(), features[0].cols, CV_32F, cv::Scalar::all(0));
    for(int i = 0; i < features.size(); i++)
    {
        features[i].copyTo(feat.row(i));
    }
    if(!online)
    {
        det->setType(cv::ml::SVM::C_SVC);
        det->setKernel(cv::ml::SVM::LINEAR);
        //det->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 1e-6));
        cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(feat,cv::ml::ROW_SAMPLE, labels);
        det->train(tData);
        cv::Mat svidx;
        b = det->getDecisionFunction(0, w, svidx);
        cv::Mat sv = det->getSupportVectors();
        w = -sv;
        w = w.t();
        //std::cout<<b<<std::endl;
        //std::cout<<feat.rows - cv::sum((((feat*w + b) > 0) & (labe > 0)) | (((feat*w + b) < 0) & (labe < 0)) )[0]/255<<std::endl;
    }
    else{
        cv::Mat feat_and_sv, osv = det->getUncompressedSupportVectors();
        cv::vconcat(feat, osv, feat_and_sv);
        cv::Mat svlabel = osv*w + b;
        for(int i = 0; i < svlabel.rows; i++)
            labels.push_back(svlabel.at<float>(i, 0) > 0 ? 1 : -1) ;
        cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(feat_and_sv,cv::ml::ROW_SAMPLE, labels);
        det->clear();
        det->train(tData);
        b = det->getDecisionFunction(0, w, svidx);
        cv::Mat sv = det->getSupportVectors();
        w = -sv;
        w = w.t();
        //std::cout<<cv::sum((((feat*w + b) < 0) & (labe > 0)) )[0]/255<<std::endl;
    }
}

void detector::hq_train(std::vector<cv::Mat> &hightquality, std::vector<int> &labels) {
    cv::Mat feat(hightquality.size(), hightquality[0].cols, CV_32F);
    for(int i = 0; i < hightquality.size(); i++)
        hightquality[i].copyTo(feat.row(i));
    cv::Mat feat_and_sv, osv = det->getUncompressedSupportVectors();
    cv::vconcat(feat, osv, feat_and_sv);
    cv::Mat svlabel = osv*w + b;
    for(int i = 0; i < svlabel.rows; i++)
        labels.push_back(svlabel.at<float>(i, 0) > 0 ? 1 : -1);
    cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(feat_and_sv,cv::ml::ROW_SAMPLE, labels);
    det->clear();
    det->train(tData);
    cv::Mat svidx;
    b = det->getDecisionFunction(0, w, svidx);
    cv::Mat sv = det->getSupportVectors();
    w = -sv;
    w = w.t();
}

std::pair<std::vector<cv::Mat>, std::vector<cv::Point> > detector::all_frame_search(const cv::Mat &image, float scaleFactor) {
    std::vector<cv::Mat> feats;
    std::vector<cv::Point> locs;
    std::uniform_int_distribution<unsigned > u(1, 10);
    int step = 0.5*std::min(target_sz.width, target_sz.height)*scaleFactor;//ceil((image.cols - target_sz.width * scaleFactor) * (image.rows - target_sz.height * scaleFactor) /(all_frame_window_num * all_frame_window_num + 0.f));
    for (int i = 0; i < image.rows - target_sz.height * scaleFactor; i += step)
    {
        for (int j = 0; j < image.cols - target_sz.width * scaleFactor; j += step) {
            if ( u(e) > 3)
                continue;
            cv::Mat feature = get_feature(
                    image(cv::Rect(j, i, target_sz.width * scaleFactor, target_sz.height * scaleFactor)));
            cv::resize(feature, feature, t_sz, 0, 0, cv::INTER_NEAREST);
            feature.convertTo(feature, CV_32F);
            feats.push_back(feature.reshape(1, 1));
            locs.push_back(cv::Point(j + target_sz.width * scaleFactor / 2, i + target_sz.height * scaleFactor / 2));
        }
    }
    return std::make_pair(feats, locs);
}