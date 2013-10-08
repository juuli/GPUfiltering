#include "GPUfilter.h"
#include "FilterBlock.h"

// The lenght of y must be len_x+len_h-1, otherwise apeshit
void convolutionCPU(const float* x, 
                    const float* h,
                    float* y, 
                    int len_x, 
                    int len_h) {
  int conv_len = len_x+len_h-1;
  //  c_log_msg(LOG_DEBUG, "CPU convolution, x len %d," 
  //                       "filter len %d,"
  //                       "convolution len %d", len_x, len_h, conv_len);
  //float* conv_buffer_h = (float*)calloc(conv_len, sizeof(float));
  //float* conv_buffer_x = (float*)calloc(conv_len, sizeof(float));

  //memcpy(conv_buffer_x, x, len_x);
  //memcpy(conv_buffer_h, h, len_h);

  for(int i = 0; i < conv_len; i++) {
    float sample = 0.f;
    for(int j = 0; j < i; j++) {
      sample += x[j]*h[i-j];
    }
    y[i] = sample;
  }

  //free(conv_buffer_h);
  //free(conv_buffer_x);
  return;
}


void convolutionGPU(const float* d_x, 
                    const float* d_h,
                    float* d_y, 
                    int len_x, 
                    int len_h) {

  int conv_len = len_x+len_h-1;
  dim3 block(256);
  dim3 grid((int)(conv_len/block.x)+1);
  printf("Block size %d, Grid size %d \n", block.x, grid.x);

  if(len_x < len_h)
    convolve<<<grid, block>>>(d_x, d_h, d_y, len_x, len_h, conv_len);
  else
    convolve<<<grid, block>>>(d_h, d_x, d_y, len_h, len_x, conv_len);

}

int launchConvolutionLinear(FilterBlock* filter_block) {
  // Note to self: float* ptr = thrust::raw_pointer_cast(&(thrust_vect[0]))
  return 0;
}


// This function takes the input buffer, updates the convolution buffer
//
int launch(float* h_input_buffers,
           float* h_output_buffers,
           float* d_filters,
           float* d_convolution_buffer,
           const int num_inputs,
           const int num_outputs,
           const int conv_buffer_len,
           const int buffer_len,
           const int filter_len) {

  dim3 block(512);
  dim3 grid((int)(conv_buffer_len)/block.x);

  return 0;
}

// This function is launched so that h is always > x
__global__ void convolve(const float* d_x, 
                         const float* d_h, 
                         float* d_ret, 
                         int len_x,
                         int len_h,
                         int len) {

  int idx =  blockIdx.x*blockDim.x +threadIdx.x;
  float sample = 0.f;
  float x = 0.f;
  float h = 0.f;

  // begin
  if(idx < len_x){
    for(int i = 0; i < idx; i++) {
      x = d_x[i];
      h = d_h[idx-i];
      sample += x*h;
    }
  }
  // middle
  else if(idx < len_h){  
    for(int i = 0; i < len_x; i++) {
      x = d_x[i];
      h = d_h[idx-i];
      sample += x*h;
    }
  }
  // end
  else if(idx < len) {
    //for(int i = 0; i < len-idx; i++) {
    //  x = d_x[i];
    //  h = d_h[len_h-i];
    //  sample += x*h;
    //}
  }

  d_ret[idx] = sample;
}




__global__ void appendConvolutionBuffer(float* d_convolution_buffer,
                                        float* buffer_len){}

__global__ void convolveBuffers(float* input_buffers,
                                float* output_buffers,
                                float* filters,
                                const int num_inputs,
                                const int num_outputs,
                                const int buffer_len,
                                const int filter_len) {



}

