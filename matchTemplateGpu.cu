#include "matchTemplateGpu.cuh"
#include <cuda_runtime_api.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <stdio.h>


using uchar = unsigned char;

constexpr int block_size = 16;
// __constant__ float d_templ[221 * 221];


__global__ void convertToFloatKernel(unsigned char* d_input, float* d_output, 
    const int width, const int height) {
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int index = (row * width + col) * 3;

    if (col < width && row < height) {
        d_output[index]     = (float)d_input[index] / 255.0f;
        d_output[index + 1] = (float)d_input[index + 1] / 255.0f;
        d_output[index + 2] = (float)d_input[index + 2] / 255.0f;
    }
}

__global__ void matchTemplateKernel(const uchar* d_img, const uchar* d_templ, int* d_result,
    const int img_row, const int img_col,
    const int templ_row, const int templ_col, 
    const int result_row, const int result_col) {

    const int row = blockDim.y * blockIdx.y + threadIdx.y; 
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < result_row && col < result_col) {
        long sum = 0;
        for(int block_row = 0; block_row < templ_row; ++block_row) {
            for(int block_col = 0; block_col < templ_col; ++block_col) {
                int img_idx = (row + block_row) * img_col + col + block_col;
                int templ_idx = block_row * templ_col + block_col;
                
                int diff = d_img[img_idx] - d_templ[templ_idx];
                sum += (diff * diff);
            }
        }
        d_result[row * result_col + col] = sum;
    }
}

__global__ void matchTemplateConstKernel(
    const float* d_img, const float* d_templ, float* d_result,
    const int img_row, const int img_col,
    const int templ_row, const int templ_col, 
    const int result_row, const int result_col) {

    const int row = blockDim.y * blockIdx.y + threadIdx.y; 
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < result_row && col < result_col) {
        float sum = 0.0f;
        #pragma unroll 8
        for(int block_row = 0; block_row < templ_row; ++block_row) {
            #pragma unroll 8
            for(int block_col = 0; block_col < templ_col; ++block_col) {
                int img_idx = (row + block_row) * img_col + col + block_col;
                int templ_idx = block_row * templ_col + block_col;
                
                float diff = d_img[img_idx] - d_templ[templ_idx];
                sum += (diff * diff);
            }
        }
        d_result[row * result_col + col] = sum;
    }
}
/*
void matchTemplateGpu(const cv::Mat& h_img, const cv::Mat& h_templ, std::vector<int>& diff) {
    const int img_col = h_img.cols;
    const int img_row = h_img.rows;
	const int templ_col = h_templ.cols;
    const int templ_row = h_templ.rows;
    const int diff_col = h_img.cols - h_templ.cols + 1;
    const int diff_row = h_img.rows - h_templ.rows + 1;
	unsigned char* d_img;
    // unsigned char* d_templ;
    int* d_diff;

    size_t img_size = img_col * img_row * sizeof(unsigned char);
	size_t templ_size = templ_col * templ_row * sizeof(unsigned char);
    size_t diff_size = diff_col * diff_row * sizeof(int);

    cudaMalloc((void**)&d_img, img_size);
    // cudaMalloc((void**)&d_templ, templ_size);
    cudaMalloc((void**)&d_diff, diff_size);
    cudaMemcpy(d_img, h_img.data, img_size, cudaMemcpyHostToDevice); 
    //cudaMemcpy(d_templ, h_templ.data, templ_size, cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(d_templ, h_templ.data, templ_size);
    cudaMemcpy(d_diff, diff.data(), diff_size, cudaMemcpyHostToDevice); 

    unsigned int grid_rows = (diff_col + block_size - 1) / block_size;
    unsigned int grid_cols = (diff_row + block_size - 1) / block_size;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(grid_rows, grid_cols);
    TimeCount::instance().start(); 
    // matchTemplateKernel<<<dim_grid, dim_block>>>(d_img, d_templ, d_diff, img_row, img_col, templ_row, templ_col, diff_row, diff_col);
    matchTemplateConstKernel<<<dim_grid, dim_block>>>(d_img, d_diff, img_row, img_col, templ_row, templ_col, diff_row, diff_col);
    
    thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_diff);
    thrust::device_ptr<int> iter = thrust::min_element(d_ptr, d_ptr + diff_col * diff_row);
    std::cout<<"idx: " << iter - d_ptr <<"\n";

    cudaMemcpy(diff.data(), d_diff, diff_size, cudaMemcpyDeviceToHost); 
    TimeCount::instance().printTime();

    cudaFree(d_img);
    cudaFree(d_templ);
    cudaFree(d_diff);
}

void matchTemplateGpu(const cv::Mat& h_img, const cv::Mat& h_templ, int& min_index) {
    const int img_col = h_img.cols;
    const int img_row = h_img.rows;
	const int templ_col = h_templ.cols;
    const int templ_row = h_templ.rows;
    const int diff_col = h_img.cols - h_templ.cols + 1;
    const int diff_row = h_img.rows - h_templ.rows + 1;
	unsigned char* d_img;
    // unsigned char* d_templ;
    int* d_diff;

    size_t img_size = img_col * img_row * sizeof(unsigned char);
	size_t templ_size = templ_col * templ_row * sizeof(unsigned char);
    size_t diff_size = diff_col * diff_row * sizeof(int);

    cudaMalloc((void**)&d_img, img_size);
    cudaMalloc((void**)&d_diff, diff_size);
    cudaMemcpy(d_img, h_img.data, img_size, cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(d_templ, h_templ.data, templ_size);

    unsigned int grid_rows = (diff_col + block_size - 1) / block_size;
    unsigned int grid_cols = (diff_row + block_size - 1) / block_size;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(grid_rows, grid_cols);
    TimeCount::instance().start(); 
    matchTemplateConstKernel<<<dim_grid, dim_block>>>(d_img, d_diff, img_row, img_col, templ_row, templ_col, diff_row, diff_col);
    cudaDeviceSynchronize();
    TimeCount::instance().printTime();

    thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_diff);
    thrust::device_ptr<int> iter = thrust::min_element(d_ptr, d_ptr + diff_col * diff_row);
    min_index = iter - d_ptr;
    std::cout<<"idx: " << min_index <<"\n";

    cudaFree(d_img);
    cudaFree(d_templ);
    cudaFree(d_diff);
}
*/


void matchTemplateGpu_v2(cv::Mat& h_img, cv::Mat& h_templ, std::vector<float>& diff) {
    TimeCount::instance().start(); 
    h_img.convertTo(h_img, CV_32FC1 , 1.0 / 255);
    h_templ.convertTo(h_templ, CV_32FC1 , 1.0 / 255);
    TimeCount::instance().printTime();

    const int img_col = h_img.cols;
    const int img_row = h_img.rows;
	const int templ_col = h_templ.cols;
    const int templ_row = h_templ.rows;
    const int diff_col = h_img.cols - h_templ.cols + 1;
    const int diff_row = h_img.rows - h_templ.rows + 1;
	float* d_img;
    float* d_templ;
    float* d_diff;

    size_t img_size = img_col * img_row * sizeof(float);
	size_t templ_size = templ_col * templ_row * sizeof(float);
    size_t diff_size = diff_col * diff_row * sizeof(float);

    TimeCount::instance().start(); 
    cudaMalloc((void**)&d_img, img_size);
    cudaMalloc((void**)&d_templ, templ_size);
    cudaMalloc((void**)&d_diff, diff_size);
    cudaMemcpy(d_img, h_img.data, img_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_templ, h_templ.data, templ_size, cudaMemcpyHostToDevice); 
    TimeCount::instance().printTime();

    unsigned int grid_rows = (diff_col + block_size - 1) / block_size;
    unsigned int grid_cols = (diff_row + block_size - 1) / block_size;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(grid_rows, grid_cols);
    TimeCount::instance().start(); 
    matchTemplateConstKernel<<<dim_grid, dim_block>>>(d_img, d_templ, d_diff, img_row, img_col, templ_row, templ_col, diff_row, diff_col);
    cudaMemcpy(diff.data(), d_diff, diff_size, cudaMemcpyDeviceToHost); 
    TimeCount::instance().printTime();

    cudaFree(d_img);
    cudaFree(d_templ);
    cudaFree(d_diff);
}
