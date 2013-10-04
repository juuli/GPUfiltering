#ifndef FILTER_BLOCK_H
#define FILTER_BLOCK_H

#include <vector>
#include <thrust/device_vector.h>
#include "GPUfilter.h"

class FilterBlock {
public:
  FilterBlock()
  : num_inputs_(1),
    num_outputs_(2),
    filter_len_(48000),
    buffer_len_(512)
  {};

  ~FilterBlock(){};

private:
  int num_inputs_;
  int num_outputs_;
  int filter_len_;
  int buffer_len_;

public:
  void assignIR(std::vector<float> ir,);

};


#endif
