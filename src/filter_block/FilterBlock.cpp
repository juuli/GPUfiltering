#include "FilterBlock.h"
#include "GPUfilter.h"

void FilterBlock::initialize() { 
  this->allocateFilters();
  cudaSetDevice(0);
  cudaDeviceReset();
  this->initializeConvolver();
}

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

void FilterBlock::allocateFilters() {
  log_msg<LOG_INFO>(L"FilterBlock::allocateFilters() - Allocating");
  for(int i = 0; i < this->getNumCombinations(); i++)
    this->filter_taps_.at(i) = std::vector<float>(this->getFilterLen(), 0.f);

  for(int i = 0; i < this->getNumInputs(); i++)
    this->delay_lines_.at(i) = std::vector<float>(this->getDelayLineLen(), 0.f);
  
  // If GPU, allocate thrust host vectors
  //if(this->selected_mode_ == T_GPU) {
  //  // Host
  //  this->convolver_.initialize(this->filter_taps_);
  //}
}

void FilterBlock::initializeConvolver() {
  log_msg<LOG_INFO>(L"FilterBlock::initializeConvolver");
  this->convolver_ = Convolver(this->getNumInputs(),
                               this->getNumOutputs(),
                               this->getFilterLen(),
                               this->getFrameLen());

  this->convolver_.initialize(this->filter_taps_);
  this->convolver_initialized_ = true;
}

int FilterBlock::getFilterContainerSize() {
  int ret = 0;
  ret = (this->filter_taps_.size()*this->filter_taps_.at(0).size());
  return ret;
}

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
    
  // If FIXED_FILTER is set, taps are trunctated
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
  float* ret = (float*)NULL;

  ret = &(this->filter_taps_.at(filter_idx)[0]);
  
  return ret;
}

int FilterBlock::getDelayLineIdx(int increment) {
  int idx = this->head_position_+increment;
  idx+=this->getDelayLineLen();
  idx = idx%this->getDelayLineLen();
  return idx;
}

void FilterBlock::convolveFrame(const float* input_frame, float* output_frame) {
  switch(this->selected_mode_) {
    case T_CPU:
      //printf("FrameCount: %d, HeadPos: %d \n", this->frame_count_, this->head_position_);
      this->convolveFrameCPU(input_frame, output_frame);
      this->frame_count_++;
      break;

    case T_GPU:
      this->convolveFrameGPU(input_frame, output_frame);
      break;

    default:
      return;
      break;
  }
}

void FilterBlock::convolveFrameCPU(const float* input_frame, float* output_frame) {
  int num_inputs = this->getNumInputs();
  int num_outputs = this->getNumOutputs();
  // quick tryout to just delay left
  int delay[2] = {0, 0};

  //printf("FilterBlock::convolveFrameCPU - inputs %d, outputs %d, frame_len %d, delay line %d \n",
  //        num_inputs, num_outputs, this->getFrameLen(), this->getDelayLineLen());

  // Write frame data to the delay line
  
  for(int i = 0; i < num_inputs; i++) {
    //printf("input %d ", i);
    for(int j = 0; j < this->getFrameLen(); j++) {
      this->delay_lines_.at(i).at(this->head_position_+j) = input_frame[j*num_inputs+i];
    }
  }

  for(int i = 0; i < num_outputs; i++) {
    for(int j = 0; j < this->getFrameLen(); j++) {
      if(i == 2)
        output_frame[j*num_outputs+i] = this->delay_lines_.at(0).at(getDelayLineIdx(j-delay[0]));
      if(i == 3)
        output_frame[j*num_outputs+i] = this->delay_lines_.at(1).at(getDelayLineIdx(j-delay[1]));
    }// end frame loop
  } // end output loop
  
  // increment the head position on the delay line and take a modulo to keep it in
  this->head_position_ = this->getDelayLineIdx(this->getFrameLen());
  return;
}

void FilterBlock::frameThrough(const float* input_frame, float* output_frame) {
  int num_inputs = this->getNumInputs();
  int num_outputs = this->getNumOutputs();
  int frame_len = this->getFrameLen();
  int d_line_len = this->getDelayLineLen();
  // quick tryout to just delay left
  int delay[2] = {0, 0};
  std::vector<float> in(num_inputs*this->getFrameLen());
  std::vector<float> out(num_outputs*this->getFrameLen());

  for(int i = 0; i < num_inputs; i++) {
    for(int j = 0; j < frame_len; j++) {
      in.at(i*frame_len+j) = input_frame[j*num_inputs+i];
    }
  }
  convolver_.passBuffersThrough(&(in[0]),
                                &(out[0]));
  
  cudaDeviceSynchronize();

  /* TODO 
    Copy delaylines, filters and output buffers to device
    
    Process, copy output buffers to host 

    interlace out
  */

  for(int i = 0; i < num_outputs; i++) {
    for(int j = 0; j < frame_len; j++) {
      if(i == 2) {
        //int h_idx = 0*d_line_len+getDelayLineIdx(j-delay[0]);
        output_frame[j*num_outputs+i] = out[0*frame_len+j];

      }

      if(i == 3) {
        //int h_idx = 1*d_line_len+getDelayLineIdx(j-delay[1]);
        output_frame[j*num_outputs+i] = out[1*frame_len+j];
      }
    }// end frame loop
  } // end output loop
  
  // increment the head position on the delay line and take a modulo to keep it in
  return;
}

void FilterBlock::convolveFrameGPU(const float* input_frame, float* output_frame) {
  int num_inputs = this->getNumInputs();
  int num_outputs = this->getNumOutputs();
  int frame_len = this->getFrameLen();
  int d_line_len = this->getDelayLineLen();
  // quick tryout to just delay left
  int delay[2] = {0, 0};
  std::vector<float> in(num_inputs*this->getFrameLen());
  std::vector<float> out(num_outputs*this->getFrameLen());

  for(int i = 0; i < num_inputs; i++) {
    for(int j = 0; j < frame_len; j++) {
      in.at(i*frame_len+j) = input_frame[j*num_inputs+i];
    }
  }
  convolver_.convolveT(&(in[0]),
                       &(out[0]));
  
  cudaDeviceSynchronize();

  /* TODO 
    Copy delaylines, filters and output buffers to device
    
    Process, copy output buffers to host 

    interlace out
  */

  for(int i = 0; i < num_outputs; i++) {
    for(int j = 0; j < frame_len; j++) {
        //int h_idx = 0*d_line_len+getDelayLineIdx(j-delay[0]);
        output_frame[j*num_outputs+i] = out[i*frame_len+j];
    }// end frame loop
  } // end output loop
  
  // increment the head position on the delay line and take a modulo to keep it in
  return;
}



