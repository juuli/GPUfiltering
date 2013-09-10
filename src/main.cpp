#include "./includes/portaudio.h"
#include "./pa_class/PortAudioClass.h"
#include "./signal_block/SignalBlock.h"

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

  pa.initialize();
  // for(int i = 0; i < pa.getNumberOfDevices(); i++)
  //   pa.printDeviceInfo(i);

  // Currently fastrack is at index 3
  pa.setCurrentDevice(3);
  pa.setNumInputChannels(2);
  pa.setNumOutputChannels(2);
  pa.setCurrentOutputChannel(0);
  pa.setCurrentInputChannel(1);
  pa.setCallback(playRecCallback);
  
  pa.setFs(96e3);
  
  sb.setFs(96e3);
  sb.setFBegin(1);
  sb.setFEnd(20000);
  sb.setLength(6);

  pa.output_data_ = sb.getSweep();
   
  pa.setupCallbackBlock();
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
