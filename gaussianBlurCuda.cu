#include <opencv2/opencv.hpp>
#include <iostream>
#include "time_count.h"

constexpr unsigned int block_size = 16;
constexpr int gaussian_size = 15;
constexpr int gaussian_size_half = gaussian_size / 2;
float gaussian_array[gaussian_size * gaussian_size];
float gaussian_array_1d[gaussian_size];
__constant__ float d_gaussian_array_c[gaussian_size * gaussian_size];
__constant__ float d_gaussian_array_c_1d[gaussian_size];

__host__ void getGaussianWeight_1d(const float sigma) {
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

__host__ void getGaussianWeight(const float sigma) {
    float sum = 0.0f;
    const float sigma2 = sigma * sigma;
    const float a = 1 / (2 * 3.14159 * sigma2);
    for(int i = 0; i < gaussian_size; ++i) {
        float dy = i - gaussian_size_half;
        for (int j = 0; j < gaussian_size; ++j) {
            float dx = j - gaussian_size_half; 
            gaussian_array[i * gaussian_size + j] = a * exp(-((dx * dx + dy * dy) / (2 * sigma2)));
            sum += gaussian_array[i * gaussian_size + j];
        }
    }
    sum = 1.0f / sum;
    for(int i = 0; i < gaussian_size; ++i) {
        for (int j = 0; j < gaussian_size; ++j) {
            gaussian_array[i * gaussian_size + j] *= sum;
        }
    }
}

__host__ void gaussianBlurNaive(const cv::Mat& input_img, cv::Mat& output_img, 
                                cv::Size wsize, const double sigma = 0.5) {
    if(wsize.height % 2 == 0 || wsize.width % 2 == 0) {
		wsize.height += 1;
		wsize.width += 1;
	}

	int padding = (wsize.height - 1) / 2;
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

__global__ void gaussianBlurKernelNaive(	const unsigned char* d_input, 
											unsigned char* d_output, 
											const int height, 
											const int width,
											const int src_width,
											const int filter_size) {
    unsigned int x = blockDim.y * blockIdx.y + threadIdx.y; 
	unsigned int y = blockDim.x * blockIdx.x + threadIdx.x;
	int boundary = filter_size / 2;
	unsigned int idx = (x - boundary) * src_width + (y - boundary);
	
	if(x > (boundary - 1) && y > (boundary - 1) && x < (height - boundary / 2) && y < (width - boundary / 2)) {
		float sum = 0.0f;
		for(int i = 0; i < filter_size; ++i) {
			int cur_row = x - boundary + i;
			for(int j = 0; j < filter_size; ++j) {
				int cur_col = y - boundary + j;
				sum += d_input[cur_row * width + cur_col] * d_gaussian_array_c[i * filter_size + j];
			}
		}
		d_output[idx] = sum;
	}
}

__global__ void gaussianBlurKernelRow(	const unsigned char* d_input, 
										unsigned char* d_output_x, 
										const int height, 
										const int width,
										const int filter_size) {
	unsigned int x = blockDim.y * blockIdx.y + threadIdx.y; 
	unsigned int y = blockDim.x * blockIdx.x + threadIdx.x;
	int boundary = filter_size / 2;
	unsigned int idx = x * width + y;

	if(x < height && y < (width - boundary / 2) && y > (boundary - 1)) {
		float sum = 0.0f;
		#pragma unroll
		for(int i = 0; i < filter_size; ++i) {
			sum += d_input[x * width + y + i] * d_gaussian_array_c_1d[i];
		}
		d_output_x[idx] = sum;
	}
}

__global__ void gaussianBlurKernelCol(	const unsigned char* d_input_x, 
										unsigned char* d_output, 
										const int height, 
										const int width,
										const int src_width,
										const int filter_size) {
	unsigned int x = blockDim.y * blockIdx.y + threadIdx.y; 
	unsigned int y = blockDim.x * blockIdx.x + threadIdx.x;
	int boundary = filter_size / 2;
	unsigned int idx = (x - boundary) * src_width + (y - boundary);

	if(x > (boundary - 1) && y > (boundary - 1) && x < (height - boundary / 2) && y < (width - boundary / 2)) {
		float sum = 0.0f;
		#pragma unroll
		for(int i = 0; i < filter_size; ++i) {
			int cur_row = x - boundary;
			sum += d_input_x[(cur_row + i) * width + y] * d_gaussian_array_c_1d[i];

		}
		d_output[idx] = sum;
	}
}

// error
__global__ void gaussianBlurKernelShare(const unsigned char* d_input, 
										unsigned char* d_output, 
										const int height, 
										const int width,
										const int src_width,
										const int filter_size) {
	__shared__ unsigned char tile[block_size][block_size];
	unsigned int x = blockDim.y * blockIdx.y + threadIdx.y; 
	unsigned int y = blockDim.x * blockIdx.x + threadIdx.x;
	int boundary = filter_size / 2;
	unsigned int idx = (x - boundary) * src_width + (y - boundary);
	

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	int cur_row = x - boundary;
	int cur_col = y - boundary;
	//tile[ty][tx] = (cur_row >= 0 && cur_col >= 0 && cur_row < height && cur_col < width)? d_input[cur_row * width + cur_col]:0;
	tile[ty][tx] = (x < height && y < width)? d_input[x * width + y]:0;
	__syncthreads();

	if(x > (boundary - 1) && y > (boundary - 1) && x < (height - boundary / 2) && y < (width - boundary / 2)) {
		float sum = 0.0f;
		for(int i = 0; i < filter_size; ++i) {
			int cur_row = x - boundary + i;
			for(int j = 0; j < filter_size; ++j) {
				int cur_col = y - boundary + j;
			
				sum += tile[ty + i][tx + j] * d_gaussian_array_c[i * filter_size + j];
		
			}
		}
		d_output[idx] = sum;
	}

	
}


void cudaGaussianBlur(const cv::Mat& input_img, cv::Mat& output_img, cv::Size wsize, const float sigma) {
	if(wsize.height % 2 == 0 || wsize.width % 2 == 0) {
		wsize.height += 1;
		wsize.width += 1;
	}
	const int padding = (wsize.height - 1) / 2;
	const int filter_size = wsize.height;
	cv::Mat temp_img;
	cv::copyMakeBorder(input_img, temp_img, padding, padding, padding, padding, cv::BORDER_REPLICATE);
	output_img = cv::Mat::zeros(input_img.size(), input_img.type());
	getGaussianWeight(sigma);
	getGaussianWeight_1d(sigma);

	const int width = input_img.cols;
    const int height = input_img.rows;
	const int t_width = temp_img.cols;
    const int t_height = temp_img.rows;
	unsigned char* d_input;
    unsigned char* d_output;
	unsigned char* d_input_x;
    size_t size = width * height * sizeof(unsigned char);
	size_t t_size = t_width * t_height * sizeof(unsigned char);
    cudaMalloc((void**)&d_input, t_size);
	cudaMalloc((void**)&d_input_x, t_size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, temp_img.data, t_size, cudaMemcpyHostToDevice); 
	cudaMemcpyToSymbol(d_gaussian_array_c, gaussian_array, filter_size * filter_size * sizeof(float));
	cudaMemcpyToSymbol(d_gaussian_array_c_1d, gaussian_array_1d, filter_size * sizeof(float));

	cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
	TimeCount time;

    unsigned int grid_rows = (t_width + block_size - 1) / block_size;
    unsigned int grid_cols = (t_height + block_size - 1) / block_size;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(grid_rows, grid_cols);

	// cudaEventRecord( start, 0 );
    // gaussianBlurKernelNaive<<<dim_grid, dim_block>>>(d_input, d_output, t_height, t_width, width, filter_size);
	// cudaEventRecord( stop, 0 );
    // cudaEventSynchronize( stop );
    // cudaEventElapsedTime( &elapsedTime,start, stop );
    // std::cout<<"Cuda gaussian blur takes time: "<<elapsedTime<<"ms"<<"\n";
    // cudaMemcpy(output_img.data, d_output, size, cudaMemcpyDeviceToHost);
    // cv::imwrite("../img_blur_cuda_naive.jpg", output_img);

	cudaEventRecord( start, 0 );
    gaussianBlurKernelRow<<<dim_grid, dim_block>>>(d_input, d_input_x, t_height, t_width, filter_size);
	cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime,start, stop );
    std::cout<<"Cuda X gaussian blur takes time: "<<elapsedTime<<"ms"<<"\n";
	cudaDeviceSynchronize();

	cudaEventRecord( start, 0 );
    gaussianBlurKernelCol<<<dim_grid, dim_block>>>(d_input_x, d_output, t_height, t_width, width, filter_size);
	cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime,start, stop );
    std::cout<<"Cuda Y gaussian blur takes time: "<<elapsedTime<<"ms"<<"\n";
    cudaMemcpy(output_img.data, d_output, size, cudaMemcpyDeviceToHost);
    cv::imwrite("../img_blur_cuda_opt.jpg", output_img);

	// cudaEventRecord( start, 0 );
    // gaussianBlurKernelShare<<<dim_grid, dim_block>>>(d_input, d_output, t_height, t_width, width, filter_size);
	// cudaEventRecord( stop, 0 );
    // cudaEventSynchronize( stop );
    // cudaEventElapsedTime( &elapsedTime,start, stop );
    // std::cout<<"Cuda share memory gaussian blur takes time: "<<elapsedTime<<"ms"<<"\n";
    // cudaMemcpy(output_img.data, d_output, size, cudaMemcpyDeviceToHost);
    // cv::imwrite("../img_blur_cuda_share.jpg", output_img);

	cudaFree(d_input);
    cudaFree(d_output);
	cudaFree(d_input_x);
}

int main() {
	cv::Mat img = cv::imread("/home/pjj/cppcode/opencv/filter/1.jpg", 0);
	int width = img.cols;
    int height = img.rows;
	cv::Mat out_img;
	cv::Mat out_img1;
	cv::Mat out_img2;
    TimeCount time;

	time.start();
	cv::GaussianBlur(img, out_img, cv::Size(15, 15), 3, 3);
	auto time_count = time.getTime();
	std::cout<<"Opencv gaussian blur takes time: "<<time_count<<"ms"<<"\n";
	cv::imwrite("../img_blur_opencv.jpg", out_img);
	
	time.start();
	gaussianBlurNaive(img, out_img1, cv::Size(15, 15), 3);
	time_count = time.getTime();
	std::cout<<"Naive gaussian blur takes time: "<<time_count<<"ms"<<"\n";
	cv::imwrite("../img_blur_naive.jpg", out_img1);


    cudaGaussianBlur(img, out_img2, cv::Size(15, 15), 3);

	
	return 0;
}