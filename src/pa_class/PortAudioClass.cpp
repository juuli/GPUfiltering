#include "PortAudioClass.h"
#include "../signal_block/SignalBlock.h"
//static FilterBlock* filter_block;

// PortAudio error check routine
bool paCheck(PaError err) {
  if(err != 0) {
    log_msg<LOG_ERROR>(L"PortAudio error num: %d, message: %s") 
                       % err % Pa_GetErrorText(err);
    return false;
  }
  return true;
}

bool PortAudioClass::initialize() {
  log_msg<LOG_INFO>(L"PortAudioClass::initialize() - port audio initialize");
  bool ret = false;
  ret = paCheck(Pa_Initialize());

  this->setInitialized(ret);
  this->number_of_devices_ = Pa_GetDeviceCount();
  this->pa_version_ = Pa_GetVersion();
  this->callback_ = defaultCallback;

  return ret;
}

bool PortAudioClass::terminate() {
  bool ret = false;
  if(!this->isInitialized()) {
    log_msg<LOG_INFO>(L"PortAudioClass::terminate() - port audio not initialized");
    return true;      
  }

  log_msg<LOG_INFO>(L"PortAudioClass::terminate() - port audio terminate");
  ret = paCheck(Pa_Terminate());
  this->setInitialized(ret);
  return ret;
}

void PortAudioClass::printInfoToLog() {
  log_msg<LOG_INFO>(L"PortAudioClass::printInfoToLog\n"
                    "  - Port Audio version: %d \n"
                    "  - Number of Devices: %d")
                    %this->getPaVersion() % this->getNumberOfDevices();
}

void PortAudioClass::setCurrentDevice(int device_idx) {
  if(device_idx >= this->getNumberOfDevices()) {
    log_msg<LOG_INFO>(L" PortAudioClass::setCurrentDevice(int device_idx) "
                      "- invalid device_idx %d"
                      " number of devices %d")
                      %device_idx %this->getNumberOfDevices();
    return;
  }

  this->current_device_ = device_idx;
  this->current_device_info_ = (PaDeviceInfo*)Pa_GetDeviceInfo(device_idx);
}

bool PortAudioClass::openStream() {
  bool ret = false;	
  log_msg<LOG_INFO>(L"PortAudioClass::openStream()");

  PaStreamParameters input_params, output_params;

  input_params.device = this->getCurrentDevice();
  input_params.channelCount = this->getNumInputChannels();
  input_params.sampleFormat = paFloat32;
  input_params.hostApiSpecificStreamInfo = NULL;
  input_params.suggestedLatency = this->current_device_info_->defaultLowInputLatency;

  output_params.device = this->getCurrentDevice();
  output_params.channelCount = this->getNumOutputChannels();
  output_params.sampleFormat = paFloat32;
  output_params.hostApiSpecificStreamInfo = NULL;
  output_params.suggestedLatency = this->current_device_info_->defaultLowOutputLatency;

  if(!paCheck(Pa_IsFormatSupported(&input_params, &output_params, this->getFs())))
    return false;

  ret = paCheck(Pa_OpenStream(&(this->stream_),
                              &(input_params),
                              &(output_params),
                              this->getFs(),
                              this->getFramesPerBuffer(),
                              0, // paClipOff 
                              this->callback_,
                              this->callback_data_ptr_)); // user data
  return ret;
}

bool PortAudioClass::closeStream() {
  bool ret = false;	
  log_msg<LOG_INFO>(L"PortAudioClass::closeStream()");
  ret = paCheck(Pa_CloseStream(this->stream_));
  return ret;
}

bool PortAudioClass::startStream() {
  bool ret = false;
  log_msg<LOG_INFO>(L"PortAudioClass::startStream()");
  ret = paCheck(Pa_StartStream(this->stream_));
  return ret;
}

std::vector< std::string > PortAudioClass::getDeviceData(int device_idx) {
  std::vector< std::string > ret;

  PaDeviceInfo *device_info = (PaDeviceInfo*)NULL;

  if(device_idx >= this->getNumberOfDevices()) {
    log_msg<LOG_INFO>(L"PortAudioClassp::printDeviceInfo(int device_idx) - invalid device_idx %d"
                      " number of devices %d")%device_idx %this->getNumberOfDevices();
    return ret;
  }
  std::stringstream ss;
  device_info = (PaDeviceInfo*)Pa_GetDeviceInfo(device_idx);
  // Name
  ss<<device_info->name;
  ret.push_back(std::string(ss.str()));
  ss.str(std::string());
  // Num inputs
  ss<<device_info->maxInputChannels;
  ret.push_back(std::string(ss.str()));
  ss.str(std::string());

  // Num outputs
  ss<<device_info->maxOutputChannels;
  ret.push_back(std::string(ss.str()));
  ss.str(std::string());
}

void PortAudioClass::printDeviceInfo(int device_idx) {
  PaDeviceInfo *device_info = (PaDeviceInfo*)NULL;

  if(device_idx >= this->getNumberOfDevices()) {
    log_msg<LOG_INFO>(L"PortAudioClassp::printDeviceInfo(int device_idx) - invalid device_idx %d"
                      " number of devices %d")%device_idx %this->getNumberOfDevices();
    return;
  }
  
  device_info = (PaDeviceInfo*)Pa_GetDeviceInfo(device_idx);
  log_msg<LOG_INFO>(L"PortAudioClassp::printDeviceInfo(int device_num) - printing info device idx %d") %device_idx;

  printf( "Name = %s\n", device_info->name );
	printf( "Host API  = %s\n",  Pa_GetHostApiInfo( device_info->hostApi )->name );
	printf( "Max inputs = %d", device_info->maxInputChannels  );
	printf( "Max outputs = %d\n", device_info->maxOutputChannels  );
	printf( "Default low input latency   = %8.4f\n", device_info->defaultLowInputLatency);
	printf( "Default low output latency  = %8.4f\n", device_info->defaultLowOutputLatency);
	printf( "Default high input latency  = %8.4f\n", device_info->defaultHighInputLatency);
	printf( "Default high output latency = %8.4f\n", device_info->defaultHighOutputLatency);
	printf( "Default sample rate         = %8.2f\n", device_info->defaultSampleRate );     
}

void PortAudioClass::setCurrentInputChannel(int channel_idx) {
  if(this->getNumOutputChannels() < channel_idx) {
    log_msg<LOG_INFO>(L"PortAudioClass::setCurrentInputChannel - "
                      "channel_idx %d not set, number of channels")
                      %channel_idx %this->getNumInputChannels();
    return;
  }

  this->current_input_channel_ = channel_idx;
}

void PortAudioClass::setCurrentOutputChannel(int channel_idx) {
  if(this->getNumOutputChannels() < channel_idx) {
    log_msg<LOG_INFO>(L"PortAudioClass::setCurrentOutputChannel - "
                      "channel_idx %d not set, number of channels")
                      %channel_idx %this->getNumOutputChannels();
    return;
  }

  this->current_output_channel_ = channel_idx;
}

void PortAudioClass::setFramesPerBuffer(unsigned long fpb) {
  this->frames_per_buffer_ = fpb;
}

CallbackStruct PortAudioClass::setupSweepCallbackBlock() {
  int data_size = this->output_data_.size();
  int num_frames = data_size/this->getFramesPerBuffer();

  if(data_size%this->getFramesPerBuffer() != 0) {
    int reminder = data_size-(num_frames*this->getFramesPerBuffer());
    reminder = this->getFramesPerBuffer()-reminder;
    num_frames += 1;
    data_size += reminder;
    this->output_data_.resize(data_size, 0.f);
  }

  // Allocate data for output and input buffer
  this->input_buffer_.assign(data_size, 0.f);

  //// This would be handy to send different data to different channels
  // ON HOLD
  //for(int i = 0; i < data_size; i++) {
  //  for(int j = 0; j < this->getNumOutputChannels(); j++) {
  //    int idx = this->getNumOutputChannels()*i+j;
  //    this->output_buffer_.at(idx) = this->output_data_.at(i);
  //  }
  // }

  CallbackStruct callback_data_;

  callback_data_.output_samples = &(this->output_data_[0]);
  callback_data_.input_samples = &(this->input_buffer_[0]);
  callback_data_.current_frame = 0;
  callback_data_.frames_left = num_frames;
  callback_data_.output_channel = this->getCurrentOutputChannel();
  callback_data_.input_channel = this->getCurrentInputChannel();
  callback_data_.num_input_channels = this->getNumInputChannels();
  callback_data_.num_output_channels = this->getNumOutputChannels();
  log_msg<LOG_INFO>(L"PortAudioClass::setupSweepCallbackBlock - \n"
                    "____________________________\n"
                    "Number of frames assigned %d \n"
                    "Output Channel %d \n"
                    "Input Channel %d \n" 
                    "____________________________\n")
                    % num_frames % this->getCurrentOutputChannel()
                    % this->getCurrentInputChannel();

  return callback_data_;
}

void PortAudioClass::setCallback(PaClassCallback callback) {
  log_msg<LOG_INFO>(L"PortAudioClass::setCallback");
  this->callback_ = callback;
}

std::vector<float> PortAudioClass::measureIR(int device_idx,
                                             int input_channel,
                                             int output_channel,
                                             float sweep_len,
                                             int num_measurements,
                                             int start_hz,
                                             int end_hz) {

  std::vector<float> ret; 
  std::vector<float> temp;

  int max_chan = input_channel > output_channel ? input_channel : output_channel;
  max_chan = 2 > max_chan ? 2 : max_chan;
  //std::cout<<"Max channels: "<<max_chan<<std::endl;
  SignalBlock sb;

  
  this->initialize();
  this->setCurrentDevice(device_idx);
  this->setFramesPerBuffer(256);
  this->setNumInputChannels(max_chan);
  this->setNumOutputChannels(max_chan);
  this->setCurrentOutputChannel(input_channel);
  this->setCurrentInputChannel(output_channel);
  
  sb.setFs(48e3);
  sb.setFBegin(start_hz);
  sb.setFEnd(end_hz);
  sb.setLength(sweep_len);
  

  
  for(int i = 0; i < num_measurements; i++) {
  this->output_data_ = sb.getSweep();
  CallbackStruct sweep = this->setupSweepCallbackBlock();
  this->setCallbackData((void*)&sweep);
  this->setCallback(playRecCallback);
  

  this->openStream();
  this->startStream();
  sleep(sb.getLength()+1);
  this->closeStream();
  
  temp = sb.getRawIr(this->getOutputData(),
                    this->getInputBuffer());

  ret.resize(temp.size(), 0.f);
  for(int j = 0; j < temp.size(); j++) {
    ret.at(j) += temp.at(j); 
  }
  }

  for(int j = 0; j < temp.size(); j++) {
    ret.at(j) /= num_measurements; 
  }

  this->terminate();
  return ret;
}