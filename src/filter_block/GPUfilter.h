#ifndef GPU_FILTER_H
#define GPU_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Forward declaration
class FilterBlock;

int setInputBuffers(float* data, int num_channels, int buffer_size);


int launchConvolutionLinear(FilterBlock* filter_block);





#endif
