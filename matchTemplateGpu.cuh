/*
 * @Author: JHC521PJJ 
 * @Date: 2023-08-02 21:06:30 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-08-02 21:06:30 
 */

#ifndef _MATCH_TEMPLATEGPU_H_
#define _MATCH_TEMPLATEGPU_H_
#include <opencv2/opencv.hpp>


void matchTemplateGpu(const cv::Mat& h_img, const cv::Mat& h_templ, std::vector<int>& diff);

void matchTemplateGpu(const cv::Mat& h_img, const cv::Mat& h_templ, int& min_index);

void matchTemplateGpu_v2(cv::Mat& h_img, cv::Mat& h_templ, std::vector<float>& diff);


#endif