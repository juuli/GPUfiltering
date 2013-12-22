#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "../pa_class/PortAudioClass.h"


// Test the basic setup block
BOOST_AUTO_TEST_CASE(PaClass_basics) {
 PortAudioClass pa;
 pa.initialize();
 std::vector<std::string> temp;
 temp = pa.getDeviceData(3);

 std::cout<<temp.at(0)<<std::endl;
 std::cout<<temp.at(1)<<std::endl;
 std::cout<<temp.at(2)<<std::endl;
 pa.terminate();
}

// Test the basic setup block
BOOST_AUTO_TEST_CASE(PaClass_sweep) {
 PortAudioClass pa;

 std::vector<float> resp;
 int device_idx = 3;
 int input = 0;
 int output = 0;
 float len = 3;
 int num = 1;
 int start_hz = 1;
 int end_hz = 20000;
 resp = pa.measureIR(device_idx, 
                     input, 
                     output, 
                     len, 
                     num, 
                     start_hz, 
                     end_hz);

}