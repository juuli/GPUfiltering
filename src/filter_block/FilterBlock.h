#ifndef FILTER_BLOCK_H
#define FILTER_BLOCK_H

#include <vector>
#include <thrust/device_vector.h>
#include "GPUfilter.h"

#define INIT_F_LEN  512*4;

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

  // Filter taps for each input/output combination
  std::vector< std::vector<float> > filter_taps_;
  int getNumCombinations(){return this->num_inputs_*this->num_outputs_;};
  void allocateFilters() {
    for(int i = 0; i < this->getNumCombinations(); i++)
      this->filter_taps_.at(i) = std::vector<float>(this->getFilterLen(), 0.f);};

public:
  void setNumInAndOutputs(int num_inputs, int num_outputs);

  int getFilterIndex(int input, int output);
  void setFilterLen(int filter_len) {this->filter_len_ = filter_len;};
  int getFilterLen() {return this->filter_len_;};
  void setFilterTaps(int input, int output, std::vector<float>& taps);
  float* getFilterTaps(int input, int output);
};


#endif
