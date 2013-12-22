#ifndef GPU_FILTER_H
#define GPU_FILTER_H

#include <limits.h>
#include <cufft.h>
#include <fftw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cudaUtils.h"

#define BUFFER_LEN 256
#define FILTER_LEN 1024*2*2*2*2*2*2*2
#define MOD(x, y) ((x+y)%y) 
#define INC_TO_MOD(x, y) ((x)+((y)-(MOD(x,y))))


// Forward declaration
class FilterBlock;

class Convolver {
public:
  Convolver() 
  : num_inputs_(1),
    num_outputs_(1),
    filter_len_(FILTER_LEN),
    buffer_len_(BUFFER_LEN),
    conv_len_(BUFFER_LEN+FILTER_LEN-1),
    d_current_in_frame_(NULL),
    d_current_out_frame_(NULL),
    d_filters_(NULL),
    d_output_dl_(NULL)
  {};

  Convolver(int inputs, 
            int outputs,
            int filter_len,
            int buffer_len) 
  : num_inputs_(inputs),
    num_outputs_(outputs),
    filter_len_(filter_len),
    buffer_len_(buffer_len),
    pad_(buffer_len-1),
    conv_len_(filter_len+buffer_len-1),
    dl_len_(filter_len+2*buffer_len-1),
    dl_loc_(0),
    d_current_in_frame_(NULL),
    d_current_out_frame_(NULL),
    d_filters_(NULL),
    d_output_dl_(NULL)
  {};

  ~Convolver() {};

private:
  int num_inputs_;
  int num_outputs_;
  int filter_len_;
  int buffer_len_;
  int pad_;
  int conv_len_;
  int dl_len_;
  int dl_loc_;
  
  float* d_current_in_frame_;
  float* d_current_out_frame_;
  float* d_filters_;
  float* d_output_dl_;

public:
  void initialize(std::vector< std::vector<float> >& filters);
  void processBuffers(float* input_buffers, 
                      float* output_buffers);

  void passBuffersThrough(float* input_buffers,
                          float* output_buffers);

  void convolveT(float* input_buffers,
                 float* output_buffers);

  void cleanup() {
    if(d_current_out_frame_) destroyMem<float>(d_current_out_frame_);
    if(d_current_in_frame_) destroyMem<float>(d_current_in_frame_);
    if(d_filters_) destroyMem<float>(d_filters_);
    if(d_output_dl_) destroyMem<float>(d_output_dl_);
  }

  float* getDFilters() {
    return this->d_filters_;
  }

};


void convolutionCPU(const float* x, 
                    const float* h, 
                    float* y,
                    int len_x, 
                    int len_h); 

void convolutionCPUfftw(fftw_complex* h_x, 
                        fftw_complex* h_h,
                        fftw_complex* h_y, 
                        fftw_plan& p,
                        int len_x, 
                        int len_h);
 
void convolutionGPUpadH(const float* d_x, 
                    const float* d_h,
                    float* d_y, 
                    int len_x, 
                    int len_h); 

void convolutionGPU(const float* x, 
                    const float* h, 
                    float* y,
                    int len_x, 
                    int len_h); 

void convolutionGPUshared(const float* x, 
                          const float* h, 
                          float* y,
                          int len_x, 
                          int len_h); 



int setInputBuffers(float* data, int num_channels, int buffer_size);


int launchConvolutionLinear(FilterBlock* filter_block);

void multiplyAddBuffers(const cufftComplex* fdl,
                        const cufftComplex* H,
                        cufftComplex* accum,
                        int fdl_idx,
                        int H_idx,
                        int window_len);

__global__ void passThroughKernel(float* d_input_, 
                                  float* d_output_,
                                  float* d_dl_,
                                  int num_channels_,
                                  int buffer_len,
                                  int dl_len, 
                                  int dl_loc); 


__global__ void multiplyAndAccumulate(const cufftComplex* x1,
                                      const cufftComplex* x2,
                                      cufftComplex* result,
                                      int len);

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

__global__ void convolvePadH(const float* d_x, 
                             const float* d_h, 
                             float* d_ret, 
                             int len_x,
                             int len_h,
                             int len);

template < int BUFFER_SIZE >
__global__ void convolvePadHShared(const float* d_x, 
                                   const float* d_h, 
                                   float* d_ret, 
                                   int len_x,
                                   int len_h,
                                   int len); 

template <int BUFFER_SIZE, int NUM_CHANNELS>
__global__ void convolveBuffers(float* d_input, 
                                float* d_output,
                                float* d_dl,
                                float* d_filters,
                                int num_channels,
                                int buffer_len,
                                int filter_len,
                                int dl_len, 
                                int dl_loc,
                                int conv_len,
                                float* debug);

#endif
