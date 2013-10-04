#include "./includes/portaudio.h"
#include "./pa_class/PortAudioClass.h"
#include "./signal_block/SignalBlock.h"
#include "./filter_block/FilterBlock.h"

#include <iostream>
#include <cmath>

void writeFile(std::string fp, 
               const std::vector<float>& sweep,
               const std::vector<float>& measured,
               std::vector<float>& raw_ir) {
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

  // Global setup

  int num_inputs = 2;
  int num_outputs = 2;

  fb.setNumInAndOutputs(num_inputs,num_outputs);

  /* initia filter setup test
  std::vector<float> filter1(10);
  filter1.at(1) = 1;  
  std::vector<float> filter2(512);
  filter2.at(480) = 1;

  fb.setFilterTaps(0,0, filter1); // left to left
  fb.setFilterTaps(1,1, filter2); // right to right
  */

  pa.initialize();
  
  for(int i = 0; i < pa.getNumberOfDevices(); i++)
    //pa.printDeviceInfo(i);

  // Currently fastrack is at index 3
  // Soundflower 16 is index 5

  // port audio setup
  pa.setCurrentDevice(5);
  pa.setNumInputChannels(2);
  pa.setNumOutputChannels(2);

  // this is for the sweep, one pair at a time
  pa.setCurrentOutputChannel(0);
  pa.setCurrentInputChannel(1);

  pa.setFs(48e3);
  
  // sweep parameters  
  sb.setFs(48e3);
  sb.setFBegin(1);
  sb.setFEnd(20000);
  sb.setLength(6);
  pa.output_data_ = sb.getSweep();

  CallbackStruct sweep = pa.setupSweepCallbackBlock();
  
  pa.setCallbackData((void*)&sweep);
  pa.setCallback(playRecCallback);

  pa.openStream();
  pa.startStream();
  std::cin.get();
  pa.closeStream();
  pa.terminate();

  std::vector<float> ir = sb.getRawIr(pa.getOutputData(),
                                      pa.getInputBuffer());
  
  writeFile("response.txt", pa.getOutputData(), pa.getInputBuffer(), ir);
  
  return 0;
}
