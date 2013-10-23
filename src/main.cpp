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
  for(unsigned int i = 0; i < len; i++) {
		respfile<<sweep.at(i)<<"  "<<measured.at(i)<<" "<<raw_ir.at(i)<<std::endl;
	}

	respfile.close();
}

int main(void) {
  loggerInit();
  PortAudioClass pa;
  SignalBlock sb;  
  FilterBlock fb;
  fb.initialize();
  // Global setup

  int num_inputs = 4;
  int num_outputs = 4;
  EXEC_MODE mode = T_GPU;

  int num_taps = 4800;
  
  std::vector<float> filter1(num_taps,0.f);
  filter1.at(0) = 1.f;

  std::vector<float> filter2(num_taps,0.f);
  filter2.at(0) = 1.f;

  fb.setFilterLen(num_taps);
  fb.setNumInAndOutputs(num_inputs,num_outputs);
  
  fb.setFilterTaps(0,2, filter1);
  fb.setFilterTaps(1,3, filter2);
  fb.setMode(mode);
  






  fb.initialize();
  // TODO put filter data in and initialize



  pa.initialize();
  
  for(int i = 0; i < pa.getNumberOfDevices(); i++)
    //pa.printDeviceInfo(i);

  // Currently fastrack is at index 3
  // Soundflower 16 is index 5

  // port audio setup
  pa.setCurrentDevice(4);
  pa.setNumInputChannels(num_inputs);
  pa.setNumOutputChannels(num_outputs);

  // this is for the sweep, one pair at a time
  pa.setCurrentOutputChannel(0);
  pa.setCurrentInputChannel(1);
  pa.setFs(48e3);
  
  // sweep parameters  
  sb.setFs(48e3);
  sb.setFBegin(1);
  sb.setFEnd(20000);
  sb.setLength(6);

  if(mode == SWEEP) {
    pa.output_data_ = sb.getSweep();
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
    writeFile("response.txt", pa.getOutputData(), pa.getInputBuffer(), ir);
  }
  return 0;
}
