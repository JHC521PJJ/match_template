/*
 * @Author: JHC521PJJ 
 * @Date: 2023-08-02 21:03:26 
 * @Last Modified by: JHC521PJJ
 * @Last Modified time: 2023-08-02 21:05:38
 */

#ifndef _MATCH_TEMPLATE_H_
#define _MATCH_TEMPLATE_H_

namespace ocr {

extern "C" __declspec(dllexport) void matchTemplate_Halocn(
    unsigned char* p_data, 
    const int img_width, 
    const int img_height, 
    const char* template_path, 
    const int boundary_row1, const int boundary_col1, 
    const int boundary_row2, const int boundary_col2, 
    const int center_x, const int center_y,
    const int angle_start, const int angle_extent, 
    const double min_score, 
    const int num_match, 
    const double max_overlap, 
    const char* sub_pixel, 
    const int num_levels, 
    const double greediness,
    int& distance, 
    int& find_flag, 
    double& score);

}

#endif
