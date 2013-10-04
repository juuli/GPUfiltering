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


  fb.setNumInAndOutputs(3,3);

  std::vector<float> filter1(10);
  filter1.at(1) = 1;
  
  std::vector<float> filter2(10);
  filter2.at(2) = 1;


  fb.setFilterTaps(0,0, filter1);
  fb.setFilterTaps(2,1, filter2);
  
  float* f_ptr = fb.getFilterTaps(2,1);
  for(int i = 0; i < 10; i++)
    printf("tap %d : %f\n", i, f_ptr[i]);

  fb.setNumInAndOutputs(4,3);
  f_ptr = fb.getFilterTaps(2,1);
  for(int i = 0; i < 10; i++)
    printf("tap %d : %f\n", i, f_ptr[i]);


  //pa.initialize();
  // for(int i = 0; i < pa.getNumberOfDevices(); i++)
  //   pa.printDeviceInfo(i);

  // Currently fastrack is at index 3
  /*
  pa.setCurrentDevice(3);
  pa.setNumInputChannels(2);
  pa.setNumOutputChannels(2);
  pa.setCurrentOutputChannel(0);
  pa.setCurrentInputChannel(1);
  pa.setCallback(playRecCallback);
  
  pa.setFs(48e3);
  
  sb.setFs(48e3);
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
  */
  return 0;
}
