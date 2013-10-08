#ifndef GPU_FILTER_H
#define GPU_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cudaUtils.h"

// Forward declaration
class FilterBlock;

void convolutionCPU(const float* x, 
                    const float* h, 
                    float* y,
                    int len_x, 
                    int len_h); 

void convolutionGPU(const float* x, 
                    const float* h, 
                    float* y,
                    int len_x, 
                    int len_h); 


int setInputBuffers(float* data, int num_channels, int buffer_size);


int launchConvolutionLinear(FilterBlock* filter_block);

__global__ void convolve(const float* d_x,  
                         const float* d_h, 
                         float* d_ret,
                         int len_x,
                         int len_h,
                         int c_len);


__global__ void convolveBuffers(float* input_buffers,
                                float* output_buffers,
                                float* filters,
                                const int num_inputs,
                                const int num_outputs,
                                const int buffer_len,
                                const int filter_len);


#endif
