/*
 * @Author: JHC521PJJ 
 * @Date: 2023-08-02 21:03:49 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-08-02 21:03:49 
 */

#include "matchTemplate.h"
#include <math.h>
#include <HalconCpp.h>
#include <HDevThread.h>


using namespace HalconCpp;

namespace ocr {

void matchTemplate_Halocn(  unsigned char*  p_data,
                        const int       img_width,
                        const int       img_height,
                        const char*     template_path,

                        const int       center_x,
                        const int       center_y,

                        const int       angle_start,
                        const int       angle_extent,
                        const double    min_score,
                        const int       num_match,
                        const double    max_overlap,
                        const char*     sub_pixel,
                        const int       num_levels,
                        const double    greediness,
                        const double    radio,

                        int& distance,
                        int& find_flag,
                        double& score) {

    HObject ho_image;
    HObject ho_model_contours;
    HTuple hv_model_id;
    HTuple hv_row, hv_col;
    HTuple hv_angle, hv_score;
    HTuple hv_length;

    GenImageInterleaved(&ho_image, Hlong(p_data), "rgb", img_width, img_height, 0, "byte", 0, 0, 0, 0, -1, 0);
    ReadShapeModel(template_path, &hv_model_id);
    GetShapeModelContours(&ho_model_contours, hv_model_id, 1);

    FindShapeModel(ho_image, hv_model_id, HTuple(angle_start).TupleRad(),
        HTuple(angle_extent).TupleRad(), min_score, num_match, max_overlap, sub_pixel,
        (HTuple(num_levels).Append(1)), greediness, &hv_row, &hv_col, &hv_angle, &hv_score);

    TupleLength(hv_score, &hv_length);
    if (hv_length.I() == 0) {
        find_flag = 0;
        score = 0.0;
    }
    else {
        find_flag = 1;
        score = hv_score[0].D();

        int center_point_y = static_cast<int>(hv_row.D());
        int center_point_x = static_cast<int>(hv_col.D());
        int y_2 = (center_point_y - center_y) * (center_point_y - center_y);
        int x_2 = (center_point_x - center_x) * (center_point_x - center_x);
        distance = static_cast<int>(sqrt(y_2 + x_2) * radio);
    }
    
}

}

