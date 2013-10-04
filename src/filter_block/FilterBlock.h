#ifndef FILTER_BLOCK_H
#define FILTER_BLOCK_H

/*  
  FilterBlock class defines a filtering
  setup which is used for filtering different inputs
  to specified outputs in real-time. 

  Interface: initialize, give an multichannel input frame, and it 
  returns an output frame which is filtered as desired

  Jukka Saarelma 4.10.2013
*/

#include <vector>
#include <fftw3.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../global_includes.h"
#include "GPUfilter.h"

#define FRAME_LEN 512
#define INIT_F_LEN  FRAME_LEN*4
#define FIXED_FILTER 1

enum process_method {T_CPU, T_GPU, FFT_CPU, FFT_GPU};

class FilterBlock {
public:
  FilterBlock()
  : head_position_(0),
    num_inputs_(1),
    num_outputs_(2),
    filter_len_(FRAME_LEN),
    frame_len_(FRAME_LEN),
    delay_line_len_(FRAME_LEN*4),
    selected_method_(T_CPU),
    filter_taps_(std::vector< std::vector<float> >(1*2)),
    delay_lines_(std::vector< std::vector<float> >(1))
  { 
    this->allocateFilters();
  };

  ~FilterBlock(){};

private:
  int head_position_;
  int num_inputs_;
  int num_outputs_;
  int filter_len_;
  int frame_len_;
  int delay_line_len_;
  process_method selected_method_;

  // Filter taps for each input/output combination
  std::vector< std::vector<float> > filter_taps_;
  std::vector< std::vector<float> > delay_lines_;

  // Device vectors, the form of these shoud be optimized
  thrust::host_vector<float> h_input_vector_;
  thrust::host_vector<float> h_output_vector_;
  thrust::host_vector<float> h_delay_lines_;
  thrust::device_vector<float> d_input_buffers_;
  thrust::device_vector<float> d_output_buffers_;
  thrust::device_vector<float> d_delay_lines_; 


  // Private methods
  int getNumCombinations(){return this->num_inputs_*this->num_outputs_;};
  void allocateFilters() {
    for(int i = 0; i < this->getNumCombinations(); i++) {
      this->filter_taps_.at(i) = std::vector<float>(this->getFilterLen(), 0.f);
      this->filter_taps_.at(i) = std::vector<float>(this->getDelayLineLen(), 0.f);}};

  void convolveFrameCPU(const float* input_frame, float* output_frame);
  void convolveFrameGPU(const float* input_frame, float* output_frame);
public:
  int getNumInputs() {return this->num_inputs_;};
  int getNumOutputs() {return this->num_outputs_;};
  void setNumInAndOutputs(int num_inputs, int num_outputs);
  int getFilterIndex(int input, int output);
  void setFilterLen(int filter_len) {this->filter_len_ = filter_len;};
  int getFilterLen() {return this->filter_len_;};
  void setFilterTaps(int input, int output, std::vector<float>& taps);
  float* getFilterTaps(int input, int output);
  int getFrameLen() {return this->frame_len_;};
  int getDelayLineIdx(int increment);
  int getDelayLineLen() {return this->delay_line_len_;};



  void convolveFrame(const float* input_frame, float* output_frame);

};


#endif
