#ifndef FILTER_BLOCK_H
#define FILTER_BLOCK_H

/*  
  FilterBlock class defines a real-time
  setup which is used for filtering different inputs
  to specified outputs. 

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

#define INIT_F_LEN  512*4;
#define FIXED_FILTER 1

class FilterBlock {
public:
  FilterBlock()
  : num_inputs_(1),
    num_outputs_(2),
    filter_len_(1024),
    buffer_len_(512),
    filter_taps_(std::vector< std::vector<float> >(1*2))
  { 
    this->allocateFilters();
  };

  ~FilterBlock(){};

private:
  int num_inputs_;
  int num_outputs_;
  int filter_len_;
  int buffer_len_;
  int delay_line_len_;

  // Filter taps for each input/output combination
  std::vector< std::vector<float> > filter_taps_;
 
  // Device vectors, the form of these shoud be optimized
  thrust::host_vector<float> h_input_vector_;
  thrust::host_vector<float> h_output_vector_;
  thrust::device_vector<float> d_input_buffers_;
  thrust::device_vector<float> d_output_buffers_;
  thrust::device_vector<float> delay_line_; 


  // Private methods
  int getNumCombinations(){return this->num_inputs_*this->num_outputs_;};
  void allocateFilters() {
    for(int i = 0; i < this->getNumCombinations(); i++)
      this->filter_taps_.at(i) = std::vector<float>(this->getFilterLen(), 0.f);};

public:
  int getNumInputs() {return this->num_inputs_;};
  int getNumOutputs() {return this->num_outputs_;};
  void setNumInAndOutputs(int num_inputs, int num_outputs);
  int getFilterIndex(int input, int output);
  void setFilterLen(int filter_len) {this->filter_len_ = filter_len;};
  int getFilterLen() {return this->filter_len_;};
  void setFilterTaps(int input, int output, std::vector<float>& taps);
  float* getFilterTaps(int input, int output);

  void convolveFrameCPU(float* input_frame, float* output_frame);
  void convolveFrameGPU(float* input_frame, float* output_frame);

};


#endif
