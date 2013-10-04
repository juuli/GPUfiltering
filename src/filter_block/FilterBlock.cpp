#include "FilterBlock.h"



void FilterBlock::setNumInAndOutputs(int num_inputs, int num_outputs) {

  log_msg<LOG_INFO>(L"FilterBlock::setNumInAndOutputs - i|o  %d|%d, past i|o %d|%d")
                    %num_inputs %num_outputs %this->num_inputs_ %this->num_outputs_;
                      
  int past_num_inputs = this->num_inputs_;
  int past_num_outputs = this->num_outputs_;
  std::vector< std::vector<float> > past_taps = this->filter_taps_;

  this->num_inputs_ = num_inputs;
  this->num_outputs_ = num_outputs;
  this->filter_taps_ = std::vector< std::vector<float> >(num_inputs*num_outputs);
  this->delay_lines_ = std::vector< std::vector<float> >(num_inputs); 
  this->allocateFilters();

  for(int i = 0; i < past_num_inputs; i++) {
    for(int j = 0; j < past_num_outputs; j++) {
      // If number of inputs/outputs are less than before, filters are ignored
      if(i < num_inputs && j < num_outputs) {
        int past_filter_idx = i*past_num_inputs+j;
        this->setFilterTaps(i,j, past_taps.at(past_filter_idx));
      }
     } // end j loop
  } // end i loop
} // end function

int FilterBlock::getFilterIndex(int input, int output) {
  return this->num_inputs_*input+output;
}

void FilterBlock::setFilterTaps(int input, int output, std::vector<float>& taps) {

  log_msg<LOG_INFO>(L"FilterBlock::setFilterTaps - in %d, out %d, num taps %d")
                    %input % output %taps.size();

  if(input > this->num_inputs_){
    log_msg<LOG_INFO>(L"FilterBlock::setFilterTaps - invalid input %d, number of inputs %d")
                    %input % this->num_inputs_;
    return;
  }
      
  if(output > this->num_outputs_) {
    log_msg<LOG_INFO>(L"FilterBlock::setFilterTaps - invalid input %d, number of inputs %d")
                      %input % this->num_inputs_;
    return;

  }

  int filter_idx = this->getFilterIndex(input, output);
  int taps_size = taps.size();
  int filter_len = this->getFilterLen();
    
  int used_taps = taps_size;
  if(FIXED_FILTER)
    used_taps = filter_len;

  if(filter_len < taps_size)
    this->filter_taps_.at(filter_idx).resize(taps_size, 0.f);

  for(int i = 0; i < taps_size; i++) {
    float tap = 0.f;
    if(taps_size > i)
      tap = taps.at(i);
   
    this->filter_taps_.at(filter_idx).at(i) = tap;
  }
}

float* FilterBlock::getFilterTaps(int input, int output) {
  int filter_idx = this->getFilterIndex(input, output);
  float* ret = &(this->filter_taps_.at(filter_idx)[0]);
  
  return ret;
}

int FilterBlock::getDelayLineIdx(int increment) {
  int idx = this->head_position_+increment;
  idx+=this->getDelayLineLen();
  idx = idx%this->getDelayLineLen();
  return idx;
}

void FilterBlock::convolveFrame(const float* input_frame, float* output_frame) {
  switch(this->selected_method_) {
    case T_CPU:
      this->convolveFrameCPU(input_frame, output_frame);
      break;

    case T_GPU:
      this->convolveFrameGPU(input_frame, output_frame);

    default:
      return;
      break;
  }
}

void FilterBlock::convolveFrameCPU(const float* input_frame, float* output_frame) {
  int num_inputs = this->getNumInputs();
  int num_outputs = this->getNumOutputs();

  // quick tryout to just delay left
  int delay[2] = {0, (100.f/48e3)};

  // Write frame data to the delay line
  for(int i = 0; i < num_inputs; i++) {
    for(int j = 0; j < this->getFrameLen(); j++) {
      this->delay_lines_.at(i).at(this->head_position_+j) = input_frame[j*num_inputs+i];
    }
  }

  for(int i = 0; i < num_outputs; i++) {
    for(int j = 0; j < this->getFrameLen(); j++) {
      output_frame[i*num_outputs+j] = this->delay_lines_.at(i).at(getDelayLineIdx(j-delay[i]));
    }
  }

  // increment the head position on the delay line and take a modulo to keep it in
  this->head_position_ = this->getDelayLineIdx(this->getFrameLen());

}

void FilterBlock::convolveFrameGPU(const float* input_frame, float* output_frame) {


}