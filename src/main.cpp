#include "./includes/portaudio.h"
#include "./pa_class/PortAudioClass.h"
#include "./signal_block/SignalBlock.h"
#include "./filter_block/FilterBlock.h"

#include <iostream>
#include <cmath>
#include <string>

void writeFile(std::string fp, 
               const std::vector<float>& sweep,
               const std::vector<float>& measured,
               std::vector<float>& raw_ir) 
{
  int len = sweep.size();
  int ir_len = raw_ir.size();
  log_msg<LOG_INFO>(L"Write File - sweep len %d, Ir len: %d") % len %ir_len;
  std::ofstream respfile(fp.c_str(),  std::fstream::out | std::fstream::trunc);
  respfile<<sweep.size()<<" ";
  for(int i = 0; i < len; i++) {
    respfile<<raw_ir.at(i)<<" ";
	}

	respfile.close();
}

void readFile(std::string fp,
              std::vector<float>& data) {
  std::ifstream respfile(fp.c_str(),  std::fstream::in);
  int size;
  int data_size = (int)data.size();
  respfile>>size;
  int num = (data_size<size ? data_size : size);
  log_msg<LOG_INFO>(L"Read File - len %d") % num;
  for(int i = 0; i < num; i++) {
    respfile>>data.at(i);
  }
  float max=0;
  for(int i = 0; i < num; i++) {
    if(std::abs(data.at(i))>max)
      max = std::abs(data.at(i));
  }

  for(int i = 0; i < num; i++) {
    data.at(i) /= max;
  }

}

int main(void) {
  loggerInit();
  PortAudioClass pa;
  SignalBlock sb;  
  FilterBlock fb;
  fb.initialize();
  
  // Global setup
  int num_inputs = 2;
  int num_outputs = 2;
  EXEC_MODE mode = T_GPU;

  int num_taps = 4800;
  
  std::vector<float> filter1(num_taps,0.f);
  filter1.at(0) = 1.f;

  std::vector<float> filter2(num_taps,0.f);
  filter2.at(0) = 1.f;

  fb.setFilterLen(num_taps);
  fb.setNumInAndOutputs(num_inputs,num_outputs);
  
  //readFile("responseL.txt", filter1);
  //readFile("responseR.txt", filter2);
  
  for(int i = 0; i < num_taps; i++)
    std::cout<<filter1.at(i)<<std::endl;

  fb.setFilterTaps(0,0, filter1);
  fb.setFilterTaps(0,1, filter2);
  //fb.setFilterTaps(1,1, filter2);
  //fb.setFilterTaps(1,0, filter2);
  fb.setFrameLen(256);
  fb.setMode(mode);

  fb.initialize();
  pa.setFramesPerBuffer(256);
  pa.initialize();
  
  for(int i = 0; i < pa.getNumberOfDevices(); i++)
    //pa.printDeviceInfo(i);

  // Currently fastrack is at index 3
  // Soundflower 16 is index 5

  // port audio setup
  pa.setCurrentDevice(3);
  pa.setNumInputChannels(num_inputs);
  pa.setNumOutputChannels(num_outputs);

 
  pa.setFs(48e3);
  
  // sweep parameters  
  if(mode == SWEEP) {
    
    pa.output_data_ = sb.getSweep();
     // this is for the sweep, one pair at a time
    pa.setCurrentOutputChannel(0);
    pa.setCurrentInputChannel(0);
    sb.setFs(48e3);
    sb.setFBegin(1);
    sb.setFEnd(20000);
    sb.setLength(6);
    CallbackStruct sweep = pa.setupSweepCallbackBlock();
    pa.setCallbackData((void*)&sweep);
    pa.setCallback(playRecCallback);
  }
  if(mode < SWEEP) {
    log_msg<LOG_INFO>(L"main - Convoltuion processing: %s")%mode_texts[(int)mode];
    pa.setCallbackData((void*)(&fb));
    pa.setCallback(convolutionCallback);
  }


  pa.openStream();
  pa.startStream();
  std::cin.get();
  pa.closeStream();
  pa.terminate();

  if(mode == SWEEP) {
    std::vector<float> ir = sb.getRawIr(pa.getOutputData(),
                                        pa.getInputBuffer());  
    writeFile("response1.txt", pa.getOutputData(), pa.getInputBuffer(), ir);
  }
  return 0;
}
