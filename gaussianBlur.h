#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

constexpr int gaussian_size = 15;
constexpr int gaussian_size_half = gaussian_size / 2;
float gaussian_array[gaussian_size][gaussian_size];
float gaussian_array_1d[gaussian_size];

void getGaussianWeight(const float sigma) {
    float sum = 0.0f;
    const float sigma2 = sigma * sigma;
    const float a = 1 / (2 * 3.14159 * sigma2);
    for(int i = 0; i < gaussian_size; ++i) {
        float dy = i - gaussian_size_half;
        for (int j = 0; j < gaussian_size; ++j) {
            float dx = j - gaussian_size_half; 
            gaussian_array[i][j] = a * exp(-((dx * dx + dy * dy) / (2 * sigma2)));
            sum += gaussian_array[i][j];
        }
    }
    sum = 1.0f / sum;
    for(int i = 0; i < gaussian_size; ++i) {
        float dy = i - gaussian_size_half;
        for (int j = 0; j < gaussian_size; ++j) {
            gaussian_array[i][j] *= sum;
        }
    }
}

void gaussianBlurNaive( const cv::Mat& input_img, cv::Mat& output_img, 
                        cv::Size wsize, const double sigma = 0.5) {
    if(wsize.height % 2 == 0 || wsize.width % 2 == 0) {
		wsize.height += 1;
		wsize.width += 1;
	}

	int padding = (wsize.height - 1) / 2;
	int wnum = wsize.height * wsize.width;
    getGaussianWeight(sigma);
	cv::Mat temp_img;
	cv::copyMakeBorder(input_img, temp_img, padding, padding, padding, padding, cv::BORDER_REPLICATE);
	output_img = cv::Mat::zeros(input_img.size(), input_img.type());

    for(int i = padding; i < input_img.rows + padding; ++i) {
		for(int j = padding; j < input_img.cols + padding; ++j) {
			float block_sum = 0.0f;	
			for(int r = 0; r < wsize.height; ++r) {
				for(int c = 0; c < wsize.height; ++c) {
					block_sum += temp_img.ptr<uchar>(i - padding + r)[j - padding + c] * 
                                 gaussian_array[r][c];
				}
			}
			output_img.ptr<uchar>(i - padding)[j - padding] = static_cast<uchar>(block_sum);
		}
	}
}

void gaussianBlurOpenmp( const cv::Mat& input_img, cv::Mat& output_img, 
                        cv::Size wsize, const double sigma = 0.5) {
    if(wsize.height % 2 == 0 || wsize.width % 2 == 0) {
		wsize.height += 1;
		wsize.width += 1;
	}

	int padding = (wsize.height - 1) / 2;
	int wnum = wsize.height * wsize.width;
    getGaussianWeight(sigma);
	cv::Mat temp_img;
	cv::copyMakeBorder(input_img, temp_img, padding, padding, padding, padding, cv::BORDER_REPLICATE);
	output_img = cv::Mat::zeros(input_img.size(), input_img.type());

#pragma omp parallel for schedule(dynamic)
    for(int i = padding; i < input_img.rows + padding; ++i) {
		for(int j = padding; j < input_img.cols + padding; ++j) {
			float block_sum = 0.0f;	
			for(int r = 0; r < wsize.height; ++r) {
				for(int c = 0; c < wsize.height; ++c) {
					block_sum += temp_img.ptr<uchar>(i - padding + r)[j - padding + c] * 
                                 gaussian_array[r][c];
				}
			}
			output_img.ptr<uchar>(i - padding)[j - padding] = static_cast<uchar>(block_sum);
		}
	}
}

// 1d optimize
void getGaussianWeight_1d(const float sigma) {
    float sum = 0.0f;
    const float sigma2 = sigma * sigma;
    const float a = 1 / (2 * 3.14159 * sigma2);
    for(int i = 0; i < gaussian_size; ++i) {
        float dx = i - gaussian_size_half;
        gaussian_array_1d[i] = a * exp(-((dx * dx) / (2 * sigma2)));
        sum += gaussian_array_1d[i];
        
    }
    sum = 1.0f / sum;
    for(int i = 0; i < gaussian_size; ++i) {
        gaussian_array_1d[i] *= sum;
    }
}

void gaussianBlur_1dOpt( const cv::Mat& input_img, cv::Mat& output_img, 
                        cv::Size wsize, const double sigma = 0.5) {
    if(wsize.height % 2 == 0 || wsize.width % 2 == 0) {
		wsize.height += 1;
		wsize.width += 1;
	}

	int padding = (wsize.height - 1) / 2;
	int wnum = wsize.height * wsize.width;
    getGaussianWeight_1d(sigma);
	cv::Mat temp_img;
    cv::Mat x_img;
	cv::copyMakeBorder(input_img, temp_img, padding, padding, padding, padding, cv::BORDER_REPLICATE);
	x_img = cv::Mat::zeros(temp_img.size(), temp_img.type());
    output_img = cv::Mat::zeros(input_img.size(), input_img.type());

    for(int i = 0; i < x_img.rows; ++i) {
		for(int j = padding; j < x_img.cols - padding; ++j) {
			float sum = 0.0f;
            for(int x = 0; x < gaussian_size; ++x) {
                sum += temp_img.ptr<uchar>(i)[j - padding + x] * gaussian_array_1d[x];
            }
            x_img.ptr<uchar>(i)[j] = static_cast<uchar>(sum);
		}
	}

    for(int i = padding; i < x_img.rows - padding; ++i) {
		for(int j = padding; j < x_img.cols - padding; ++j) {
			float sum = 0.0f;
            for(int y = 0; y < gaussian_size; ++y) {
                sum += x_img.ptr<uchar>(i - padding + y)[j] * gaussian_array_1d[y];
            }
            output_img.ptr<uchar>(i - padding)[j - padding] = static_cast<uchar>(sum);
		}
	}
}

void gaussianBlur_SSEOpt( const cv::Mat& input_img, cv::Mat& output_img, 
                        cv::Size wsize, const double sigma = 0.5) {
    if(wsize.height % 2 == 0 || wsize.width % 2 == 0) {
		wsize.height += 1;
		wsize.width += 1;
	}

	int padding = (wsize.height - 1) / 2;
	int wnum = wsize.height * wsize.width;
    getGaussianWeight_1d(sigma);
	cv::Mat temp_img;
    cv::Mat x_img;
	cv::copyMakeBorder(input_img, temp_img, padding, padding, padding, padding, cv::BORDER_REPLICATE);
	x_img = cv::Mat::zeros(temp_img.size(), temp_img.type());
    output_img = cv::Mat::zeros(input_img.size(), input_img.type());

    for(int i = 0; i < x_img.rows; ++i) {
        unsigned char* p_temp = temp_img.ptr<uchar>(i);
		for(int j = padding; j < x_img.cols - padding; ++j) {
			float sum = 0.0f;
            int idx = j - padding;
            __m256 m_img = _mm256_set_ps(   p_temp[idx + 0],
                                            p_temp[idx + 1],
                                            p_temp[idx + 2],
                                            p_temp[idx + 3],
                                            p_temp[idx + 4],
                                            p_temp[idx + 5],
                                            p_temp[idx + 6],
                                            p_temp[idx + 7]);
            __m256 m_garr = _mm256_loadu_ps(gaussian_array_1d);
            __m256 m_mul = _mm256_mul_ps(m_img, m_garr);
            float* p_sum = (float*)&m_mul;
            sum =   p_sum[0] + p_sum[1] + p_sum[2] + p_sum[3] + 
                    p_sum[4] + p_sum[5] + p_sum[6] + p_sum[7];
            for(int k = 8; k < gaussian_size; ++k) {
                sum += temp_img.ptr<uchar>(i)[j - padding + k] * gaussian_array_1d[k];
            }
            x_img.ptr<uchar>(i)[j] = static_cast<uchar>(sum);
		}
	}       

    for(int i = padding; i < x_img.rows - padding; ++i) {
		for(int j = padding; j < x_img.cols - padding; ++j) {
			float sum = 0.0f;
            int idx = i - padding;
            unsigned char *p0 = x_img.ptr<uchar>(idx);
            unsigned char *p1 = p0 + x_img.cols;
            unsigned char *p2 = p1 + x_img.cols;
            unsigned char *p3 = p2 + x_img.cols;
            unsigned char *p4 = p3 + x_img.cols;
            unsigned char *p5 = p4 + x_img.cols;
            unsigned char *p6 = p5 + x_img.cols;
            unsigned char *p7 = p6 + x_img.cols;
            __m256 m_img = _mm256_set_ps(   p0[j], p1[j],
                                            p2[j], p3[j],
                                            p4[j], p5[j],
                                            p6[j], p7[j]);
            __m256 m_garr = _mm256_loadu_ps(gaussian_array_1d);
            __m256 m_mul = _mm256_mul_ps(m_img, m_garr);
            float* p_sum = (float*)&m_mul;
            sum =   p_sum[0] + p_sum[1] + p_sum[2] + p_sum[3] + 
                    p_sum[4] + p_sum[5] + p_sum[6] + p_sum[7];

            for(int k = 8; k < gaussian_size; ++k) {
                sum += x_img.ptr<uchar>(i - padding + k)[j] * gaussian_array_1d[k];
            }
            
            output_img.ptr<uchar>(i - padding)[j - padding] = static_cast<uchar>(sum);
		}
	}         
}
