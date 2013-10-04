#include "GPUfilter.h"
#include "FilterBlock.h"

int launchConvolutionLinear(FilterBlock* filter_block) {

  // Note to self: float* ptr = thrust::raw_pointer_cast(&(thrust_vect[0]))

  return 0;
 
}


__global__ void convolveBuffers(float* input_buffers,
                                float* output_buffers,
                                float* filters,
                                const int num_inputs,
                                const int num_outputs,
                                const int buffer_len,
                                const int filter_len) {



}

