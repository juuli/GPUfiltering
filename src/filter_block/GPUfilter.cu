#include "GPUfilter.h"
//#include "FilterBlock.h"


void Convolver::initialize(std::vector< std::vector<float> >& filters) {
  c_log_msg(LOG_INFO, "Convolver::initialize");
  this->cleanup(); // Destroy past filters
  int pad = this->buffer_len_-1;
  int f_len = this->filter_len_;
  int number_of_filters = this->num_outputs_*this->num_inputs_;
  c_log_msg(LOG_INFO, "Convolver::initialize - filter len %d, pad %d",
            f_len, pad);
  int d_h_size = number_of_filters*(f_len+2*pad); // filter with padding
  int d_dl_size = this->num_outputs_*this->dl_len_; // output delay line
  int d_y_size = this->num_outputs_*this->buffer_len_; // output buffer
  int d_x_size = this->num_inputs_*this->buffer_len_; // input buffer

  float* d_h = valueToDevice<float>(d_h_size, 0.f, 0);
  float* d_dl = valueToDevice<float>(d_dl_size, 0.f, 0);
  float* d_y = valueToDevice<float>(d_y_size, 0.f, 0);
  float* d_x = valueToDevice<float>(d_x_size, 0.f, 0);

  c_log_msg(LOG_INFO, "Convolver::initialize - number of filters %d", number_of_filters);
  
  // Put filter to device, add the padding
  for(int i = 0; i < number_of_filters; i++) {
    float* h_filter_ptr = &(filters.at(i)[0]);
    float* d_filter_ptr = &(d_h[pad+i*(f_len+2*pad)]);
    copyHostToDevice<float>(f_len, d_filter_ptr, h_filter_ptr, 0);
  }

  this->d_filters_ = d_h;
  this->d_output_dl_ = d_dl;
  this->d_current_out_frame_ = d_y;
  this->d_current_in_frame_ = d_x;
  c_log_msg(LOG_INFO, "Convolver::initialize - done");
}

void Convolver::passBuffersThrough(float* input_buffers,
                                   float* output_buffers) {
  int b_l = this->num_inputs_*this->buffer_len_;
  float* d_in= this->d_current_in_frame_;
  float* d_out = this->d_current_out_frame_;
  float* d_dl = this->d_output_dl_;

  copyHostToDevice<float>(b_l, d_in, input_buffers, 0);

  dim3 block(512);
  dim3 grid((int)(this->buffer_len_/block.x)+1);

  passThroughKernel<<<grid, block>>>(d_in,
                                     d_out,
                                     d_dl,
                                     this->num_inputs_,
                                     this->buffer_len_,
                                     this->dl_len_,
                                     this->dl_loc_);

  
  copyDeviceToHost<float>(b_l, output_buffers, d_out, 0);
  this->dl_loc_ = MOD(this->dl_loc_+=this->buffer_len_, this->dl_len_);
}

void Convolver::convolveT(float* input_buffers,
                          float* output_buffers) {
  // Size of the frame
  int b_l = this->num_inputs_*this->buffer_len_;
  // Filter len, fitlers are padded
  int f_l = this->filter_len_+2*this->pad_;

  float* d_in= this->d_current_in_frame_;
  float* d_out = this->d_current_out_frame_;
  float* d_dl = this->d_output_dl_;
  float* d_h = this->d_filters_;

  // Copy current frame to device
  copyHostToDevice<float>(b_l, d_in, input_buffers, 0);
  
  float* debug = (float*)NULL;
  //float* debug = valueToDevice<float>(this->conv_len_, 0.f, 0);

  dim3 block(256);
  dim3 grid((int)(this->conv_len_/block.x)+1);
  c_log_msg(LOG_DEBUG, "Launch, block x %u, y %u, z %u, grid x %d, y %d, z %d",
            block.x, block.y, block.z, grid.x, grid.y, grid.z);

  convolveBuffers<256, 2><<<grid, block>>>(d_in,
                                          d_out,
                                          d_dl,
                                          d_h+this->pad_,
                                          this->num_inputs_,
                                          this->buffer_len_,
                                          f_l,
                                          this->dl_len_,
                                          this->dl_loc_,
                                          this->conv_len_,
                                          debug);

  
  //copyDeviceToHost<float>(b_l, output_buffers, d_out, 0);
  //float* h_debug = fromDevice<float>(this->conv_len_, debug, 0);
  //for(int i = 0; i < this->conv_len_; i++)
  //  printf("Debug vec %d: %f \n", i, h_debug[i]);

  this->dl_loc_ = MOD(this->dl_loc_+=this->buffer_len_, this->dl_len_);

  //destroyMem<float>(debug);
  //printf("Dl loc %d\n", this->dl_loc_);
}

// The lenght of y must be len_x+len_h-1, otherwise apeshit
void convolutionCPU(const float* h_x, 
                    const float* h_h,
                    float* h_y, 
                    int len_x, 
                    int len_h) {

  int conv_len = len_x+len_h-1;
  for(int global_idx = 0; global_idx < conv_len; global_idx++) {
    float sample = 0.f;
    if(global_idx < len_x) {
      for(int i = 0; i < global_idx; i++) {
        sample += h_x[i]*h_h[global_idx-i];
      }
    }
    // middle
    else if(global_idx < len_h){  
      for(int i = 0; i < len_x; i++) {
        sample += h_x[i]*h_h[global_idx-i];      
      }
    }
    // end
    else {
      int dist = conv_len-global_idx;
      for(int i = 0; i < dist; i++) {
        sample += h_x[len_x-dist+i]*h_h[len_h-1-i];
      }
    }
    h_y[global_idx] = sample;
  }
  return;
}

void convolutionCPUfftw(fftw_plan& fft,
                        fftw_plan& ifft,
                        int len_x, 
                        int len_h) {
}

void convolutionGPU(const float* d_x, 
                    const float* d_h,
                    float* d_y, 
                    int len_x, 
                    int len_h) {

  int conv_len = len_x+len_h-1;
  dim3 block(512);
  dim3 grid((int)(conv_len/block.x)+1);

  //printf("Convlen %d, filter len %d , buffer len %d "
  //       "Block size %d, Grid size %d \n", 
  //       conv_len, len_h, len_x, block.x, grid.x);

  convolve<<<grid, block>>>(d_x, d_h, d_y, len_x, len_h, conv_len);
}

void convolutionGPUpadH(const float* d_x, 
                    const float* d_h,
                    float* d_y, 
                    int len_x, 
                    int len_h) {

  int pad = len_x-1;
  int conv_len = len_x+len_h-1;
  dim3 block(512);
  dim3 grid((int)(conv_len/block.x)+1);
  //printf("PAD Convlen %d, filter len %d , buffer len %d "
  //       "Block size %d, Grid size %d \n", 
  //       conv_len, len_h, len_x, block.x, grid.x);

  convolvePadH<<<grid, block>>>(d_x, d_h+pad, d_y, len_x, len_h, conv_len);

}

void convolutionGPUshared(const float* d_x, 
                          const float* d_h,
                          float* d_y, 
                          int len_x, 
                          int len_h) {
  int pad = len_x-1;
  int conv_len = len_x+len_h-1;
  dim3 block(BUFFER_LEN);
  dim3 grid((int)(conv_len/block.x)+1);
  //printf("PAD Convlen %d, filter len %d , buffer len %d "
  //       "Block size %d, Grid size %d \n", 
  //       conv_len, len_h, len_x, block.x, grid.x);

  convolvePadHShared<BUFFER_LEN><<<grid, block>>>(d_x, d_h+pad, d_y, len_x, len_h, conv_len);
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

void multiplyAddBuffers(const cufftComplex* fdl,
                        const cufftComplex* H,
                        cufftComplex* accum,
                        int fdl_idx,
                        int H_idx,
                        int window_len) {

  int fdl_mem_pos = fdl_idx*window_len;
  int H_mem_pos = H_idx*window_len;
  dim3 block(BUFFER_LEN);
  dim3 grid(window_len/BUFFER_LEN+1);
  multiplyAndAccumulate<<<grid, block>>>(fdl+fdl_mem_pos,
                                         H+H_mem_pos,
                                         accum,
                                         window_len);
}

/////////////////////////////
// Kernels
//////

__device__ void cudMultiplyAdd(const cufftComplex* a, const cufftComplex* b, cufftComplex* c) {
  (*c).x += (*a).x*(*b).x-(*a).y*(*b).y;
  (*c).y += (*a).x*(*b).y+(*a).y*(*b).x;
}

__global__ void multiplyAndAccumulate(const cufftComplex* x1,
                                      const cufftComplex* x2,
                                      cufftComplex* result,
                                      int len) {
  int global_idx =  blockIdx.x*blockDim.x+threadIdx.x;
  if(global_idx < len)
    cudMultiplyAdd(&(x1[global_idx]), &(x2[global_idx]), &(result[global_idx]));
}

// This function is launched so that h is always > x
__global__ void convolve(const float* d_x, 
                         const float* d_h, 
                         float* d_ret, 
                         int len_x,
                         int len_h,
                         int len) {

  int global_idx =  blockIdx.x*blockDim.x +threadIdx.x;
  float sample = 0.f;
  float x = 0.f;
  float h = 0.f;

  // begin
  if(global_idx<len){
    if(global_idx < len_x) {
      for(int i = 0; i < global_idx; i++) {
        x = d_x[i];
        h = d_h[global_idx-i];
        sample += x*h;
      }
    }
  // middle
    else if(global_idx < len_h){  
      for(int i = 0; i < len_x; i++) {
        x = d_x[i];
        h = d_h[global_idx-i];
        sample += x*h;
      }
    }
  // end
    else {
    int dist = len-global_idx;
    for(int i = 0; i < dist; i++) {
      x = d_x[len_x-dist+i];
      h = d_h[len_h-1-i];
      sample += x*h;
    }
  }
  d_ret[global_idx] = sample;
  }
}

// This function is launched so that h is always > x
__global__ void convolvePadH(const float* d_x, 
                             const float* d_h, 
                             float* d_ret, 
                             int len_x,
                             int len_h,
                             int len) {
 
  int global_idx =  blockIdx.x*blockDim.x+threadIdx.x;

  float sample = 0.f;
  float x = 0.f;
  float h = 0.f;

  if(global_idx<len){
    // H padded, just run it
   //#pragma unroll
   for(int i = 0; i < len_x; i++) {
     x = d_x[i];
     h = d_h[global_idx-i];
     sample += x*h;
   }
  }
  d_ret[global_idx] = sample;
}

template < int BUFFER_SIZE >
__global__ void convolvePadHShared(const float* d_x, 
                         const float* d_h, 
                         float* d_ret, 
                         int len_x,
                         int len_h,
                         int len) {
  /// Shared memory allocation
  __shared__ int block_begin;
  block_begin =  blockIdx.x*blockDim.x;

  int thread_double = threadIdx.x+BUFFER_SIZE;
  int global_idx =  block_begin+threadIdx.x;

  __shared__ float s_h[2*BUFFER_SIZE];
  __shared__ float s_x[BUFFER_SIZE];

  s_h[threadIdx.x] = d_h[global_idx-BUFFER_SIZE+1];
  s_h[thread_double] = d_h[block_begin+thread_double-BUFFER_SIZE+1];

  s_x[threadIdx.x] = d_x[threadIdx.x];

  __syncthreads();
  /// End shared memory
    
  float sample = 0.f;

  if(global_idx<len){
    for(int i = 0; i < len_x; i++) {
      sample+=s_x[i]*s_h[BUFFER_SIZE-1+threadIdx.x-i];
    }
  }

  d_ret[global_idx] = sample; 
}


__global__ void passThroughKernel(float* d_input, 
                                  float* d_output,
                                  float* d_dl,
                                  int num_channels,
                                  int buffer_len,
                                  int dl_len, 
                                  int dl_loc) {

  int global_idx =  blockIdx.x*blockDim.x+threadIdx.x;
  int dl_idx = MOD(dl_loc+global_idx, dl_len);

  if(global_idx < buffer_len) {

    #pragma unroll
    for(int i = 0; i < num_channels; i++) {
      d_dl[dl_idx+i*dl_len] = d_input[global_idx+i*buffer_len];
    }

    #pragma unroll
    for(int j = 0; j < num_channels; j++) {
      d_output[global_idx+j*buffer_len] = d_dl[dl_idx+j*dl_len];
    }
  }
}

__global__ void appendConvolutionBuffer(float* d_convolution_buffer,
                                        float* buffer_len){}

template <int BUFFER_SIZE, int NUM_CHANNELS>
__global__ void convolveBuffers(float* d_input, 
                                float* d_output,
                                float* d_dl,
                                float* d_h,
                                int num_channels,
                                int buffer_len,
                                int filter_len,
                                int dl_len, 
                                int dl_loc,
                                int conv_len,
                                float* debug)  {
  /// Shared memory allocation
  __shared__ int block_begin;
  block_begin =  blockIdx.x*blockDim.x;

  int global_idx =  block_begin+threadIdx.x;
  int thread_double = threadIdx.x+BUFFER_SIZE;
  

  __shared__ float s_h[NUM_CHANNELS][2*BUFFER_SIZE];
  __shared__ float s_x[NUM_CHANNELS][BUFFER_SIZE];

  for(int i = 0; i < NUM_CHANNELS; i++) {
    int f_idx = i*filter_len-BUFFER_SIZE+1;
    s_h[i][threadIdx.x] = d_h[global_idx+f_idx];
    s_h[i][thread_double] = d_h[block_begin+thread_double+f_idx];
    s_x[i][threadIdx.x] = d_input[threadIdx.x+i*buffer_len];
  }
   __syncthreads();  /// End shared memory allocation
  
  // Calculate delayline indices
  int dl_idx_past = MOD(dl_loc+global_idx-buffer_len, dl_len);
  int dl_idx = MOD(dl_loc+global_idx, dl_len);


  ////////////////
  // Go through the length of the convolution
  //
  // Calculate the y values via convolution of x and h
  // of the current channel
  //
  // Grab the tail of the past convolution from the 
  // delay line
  //
  // Save values to the dealy line
  //

  if(global_idx<conv_len) {
    for(int j = 0; j < NUM_CHANNELS; j++) {    
      float sample = 0.f;
      float dl_val = d_dl[dl_idx+dl_len*j];

      for(int i = 0; i < buffer_len; i++) {
        sample+= s_x[j][i]*s_h[j][thread_double-i-1];
      }

      // Add the sample to the delay line sample
      sample += dl_val;
      // Insert the convolved sample to the delay line
      d_dl[dl_idx+dl_len*j] = sample;
      //if(j == 0)
      //  debug[global_idx] = sample;

      // Insert the sample to the output buffer
      if(global_idx < buffer_len){
        // Erase past buffer in the delay line
        d_dl[dl_idx_past+dl_len*j] = 0.f;
        //sample += s_x[j][global_idx];
        //d_dl[dl_idx+dl_len*j] = sample;

        d_output[global_idx+j*buffer_len] = sample;

      }
    }
    
    }
}

