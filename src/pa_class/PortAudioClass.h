#ifndef PORT_AUDIO_CLASS_H
#define PORT_AUDIO_CLASS_H

#include "../global_includes.h"
#include "../includes/portaudio.h"
#include "../filter_block/FilterBlock.h"
#include <fstream>
#include <string>

// Error printer function
bool paCheck(PaError err);

typedef int (*PaClassCallback)(const void *inputBuffer, void *outputBuffer,
                               unsigned long framesPerBuffer,
                               const PaStreamCallbackTimeInfo* timeInfo,
                               PaStreamCallbackFlags statusFlags,
                               void *userData);

typedef struct callback_struct_t{
  float* output_samples;
  float* input_samples;
  int current_frame;
  int frames_left;
  int output_channel;
  int input_channel;
  int num_input_channels;
  int num_output_channels;
} CallbackStruct;


static int convolutionCallback(const void *input_buffer, void *output_buffer,
                               unsigned long frames_per_buffer,
                               const PaStreamCallbackTimeInfo* timeInfo,
                               PaStreamCallbackFlags statusFlags,
                               void *user_data ) {
  //clock_t start_t;
  //clock_t end_t;
  //start_t = clock();  
  FilterBlock* fb = (FilterBlock*)user_data;
  //fb->frameThrough((const float*)input_buffer, (float*)output_buffer);
  fb->convolveFrameGPU((const float*)input_buffer, (float*)output_buffer);
  //end_t = clock()-start_t;
  //log_msg<LOG_INFO>(L"Convolve kernel - time: %f ms") 
  //                  % ((float)end_t/CLOCKS_PER_SEC*1000.f);
  
  return paContinue;
}

static int playRecCallback(const void *inputBuffer, void *outputBuffer,
                           unsigned long frames_per_buffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *user_data ) {
  float *out = (float*)outputBuffer;
  const float *in = (const float*)inputBuffer;

  CallbackStruct* host_data = (CallbackStruct*)user_data;
    
  int start_idx = host_data->current_frame*frames_per_buffer;

  (void) timeInfo; /* Prevent unused variable warnings. */
  (void) statusFlags;

  if(inputBuffer == NULL || host_data->frames_left==0) {
    for(unsigned int i=0; i<frames_per_buffer; i++ ) {
      int idx = i*host_data->num_output_channels+host_data->output_channel;
      out[idx] = 0.f;  /* left - silent */
    }
  }
  else {
    for(unsigned int i=0; i<frames_per_buffer; i++ ) {
      int output_idx = i*host_data->num_input_channels+host_data->output_channel;
      int input_idx = i*host_data->num_input_channels+host_data->input_channel;
      out[output_idx] = host_data->output_samples[start_idx+i];
      host_data->input_samples[start_idx+i] = in[input_idx];
    }

    host_data->current_frame +=1;
    host_data->frames_left -= 1;
  }
  return paContinue;
};

// Callback which fetches input samples and passes them to output
static int inputdataCallback(const void *inputBuffer, void *outputBuffer,
                             unsigned long frames_per_buffer,
                             const PaStreamCallbackTimeInfo* timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void *user_data)
{
  float *out = (float*)outputBuffer;
  const float *in = (const float*)inputBuffer;
	(void) statusFlags;
  (void) timeInfo; /* Prevent unused variable warnings. */
  
  CallbackStruct* output_data = (CallbackStruct*)user_data;
  
  if(inputBuffer == NULL) {
      for(unsigned int i=0; i<frames_per_buffer; i++) {
       int idx = i*output_data->num_output_channels+output_data->output_channel;
       out[idx] = 0.f;  /* left - silent */
	    }
  }
  else {
      for(unsigned int i=0; i<frames_per_buffer; i++) {
        int output_idx = i*output_data->num_input_channels+output_data->output_channel;
        int input_idx = i*output_data->num_input_channels+output_data->input_channel;
        out[output_idx] = in[input_idx]*0.9f;
      }
  }
  return paContinue;
};

static int outputdataCallback(const void *inputBuffer, void *outputBuffer,
                              unsigned long frames_per_buffer,
                              const PaStreamCallbackTimeInfo* timeInfo,
                              PaStreamCallbackFlags statusFlags,
                              void *user_data )
{
  float *out = (float*)outputBuffer;
  (void*)inputBuffer;

  CallbackStruct* output_data = (CallbackStruct*)user_data;
  
  int start_idx = output_data->current_frame*frames_per_buffer;
  //start_idx *= output_data->num_output_channels;

  (void) timeInfo; /* Prevent unused variable warnings. */
  (void) statusFlags;


  if(inputBuffer == NULL || output_data->frames_left==0) {
      for(unsigned int i=0; i<frames_per_buffer; i++ ) {
       int idx = i*output_data->num_output_channels+output_data->output_channel;
       out[idx] = 0.f;  /* left - silent */
      }
  }
  else {
      for(unsigned int i=0; i<frames_per_buffer; i++ ) {
        int idx = i*output_data->num_output_channels+output_data->output_channel;
        out[idx] = output_data->output_samples[start_idx+i]*0.9f;
      }
      output_data->current_frame +=1;
      output_data->frames_left -= 1;
  }
  return paContinue;
};

static int defaultCallback(const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData )
{
  float *out = (float*)outputBuffer;
  (void*)inputBuffer;
	
  unsigned int i;
  (void) timeInfo; /* Prevent unused variable warnings. */
  (void) statusFlags;
  (void) userData;

  if( inputBuffer == NULL ) {
      for( i=0; i<framesPerBuffer; i++ ) {
          out[i*0] = 0;  /* left - silent */
          out[i*0+1] = 0;  /* right - silent */
      }
  }
  else {
      for( i=0; i<framesPerBuffer; i++ ) {
      }
  }
  
  return paContinue;
};


class PortAudioClass {
public:
  // Constructors and initialization
  PortAudioClass()
  : output_data_(),
    initialized_(false),
    number_of_devices_(0),
    pa_version_(0),
    current_device_(-1),
    fs_(48e3),
    frames_per_buffer_(512),
    num_input_channels_(0),
    num_output_channels_(0),
    current_input_channel_(0),
    current_output_channel_(0),
    callback_data_ptr_((void*)NULL),
    output_buffer_(),
    input_buffer_(),
    stream_(NULL),
    current_device_info_(NULL),
    callback_(NULL)
  {};

  ~PortAudioClass() {};

  bool initialize();
  bool terminate();
  std::vector<float> output_data_;
private:
  bool initialized_;
  int number_of_devices_;
  int pa_version_;
  int current_device_;
  int fs_;
  unsigned long frames_per_buffer_;
  // Channel configuration
  // Currently 1 in, 1 out
  int num_input_channels_;
  int num_output_channels_;
  int current_input_channel_;
  int current_output_channel_;

  // Data which is passed to the callback function
  void* callback_data_ptr_;

  std::vector<float> output_buffer_;
  std::vector<float> input_buffer_;
  
  PaStream* stream_;
  PaDeviceInfo* current_device_info_;
  PaClassCallback callback_;


public:
  const std::vector<float>& getOutputData() {return this->output_data_;};
  const std::vector<float>& getInputBuffer() {return this->input_buffer_;};
  const std::vector<float>& getOutputBuffer() {return this->output_buffer_;};
  void printInfoToLog();
  void printDeviceInfo(int device_num);
  void setInitialized(bool initialized) {this->initialized_ = initialized;};
  bool isInitialized() {return this->initialized_;};
  int getNumberOfDevices() {return this->number_of_devices_;};
  int getPaVersion() {return this->pa_version_;};
  int getCurrentDevice() {return this->current_device_;};  
  int getFs() {return this->fs_;};
  unsigned long getFramesPerBuffer() {return this->frames_per_buffer_;};
  int getNumInputChannels() {return this->num_input_channels_;};
  int getNumOutputChannels() {return this->num_output_channels_;};
  int getCurrentInputChannel() {return this->current_input_channel_;};
  int getCurrentOutputChannel() {return this->current_output_channel_;};
  void setCurrentDevice(int device);
  void setFs(int fs) {this->fs_ = fs;};
  void setFramesPerBuffer(unsigned long fpb);
  void setNumInputChannels(int num_channels)
    {this->num_input_channels_ = num_channels;};
  void setNumOutputChannels(int num_channels)
    {this->num_output_channels_ = num_channels;};
  void setCurrentInputChannel(int channel_idx);
  void setCurrentOutputChannel(int channel_idx);
  void setCallback(PaClassCallback callback);
  void setCallbackData(void* data_ptr) {this->callback_data_ptr_ = data_ptr;};
  bool openStream();  
  bool closeStream();
  bool startStream();
  CallbackStruct setupSweepCallbackBlock();
  bool setupFilterCallback();
};


#endif
