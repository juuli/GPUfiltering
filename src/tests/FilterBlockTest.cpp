#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "../filter_block/FilterBlock.h"

BOOST_AUTO_TEST_CASE(FilterBlock_construct) {
  FilterBlock fb;

  BOOST_CHECK_EQUAL(fb.getFrameLen(), 512);
}

// Test the basic setup block
BOOST_AUTO_TEST_CASE(FilterBlock_set_taps) {
  FilterBlock fb;
  int num_taps = 100;
  fb.setNumInAndOutputs(4,4);
  BOOST_CHECK_EQUAL(fb.getFilterContainerSize(), 4*4*fb.getFilterLen());
  
  std::vector<float> filter1(num_taps,0.f);
  filter1.at(2) = 1.f;

  std::vector<float> filter2(num_taps,0.f);
  filter2.at(4) = 2.f;

  fb.setFilterTaps(1,3, filter1);
  fb.setFilterTaps(0,0, filter2);
  fb.initialize();

  printf("Filter cont size %zu\n", fb.getFilterContainerSize());
  fb.getFilterTaps(0,0);
  float* tap_ptr = (float*)NULL;
  tap_ptr = fb.getFilterTaps(1,3);
  float* tap_ptr2 = NULL;
  tap_ptr2 = fb.getFilterTaps(0,0);

  bool check = true;

  for(int i = 0; i < fb.getFilterLen(); i++) {
    if(i < num_taps) {
      if(tap_ptr[i] != filter1.at(i)) {
        check = false;
      }
    }
    else {
      if(tap_ptr[i] != 0.f)
          check = false;
    }
  }

  BOOST_CHECK_EQUAL(check, true);

  fb.setNumInAndOutputs(2,4);
  BOOST_CHECK_EQUAL(fb.getFilterContainerSize(), 2*4*fb.getFilterLen());
  tap_ptr = fb.getFilterTaps(1,3);

  for(int i = 0; i < fb.getFilterLen(); i++) {
    if(i < num_taps) {
      if(tap_ptr[i] != filter1.at(i)) {
        check = false;
      }
    }
    else {
      if(tap_ptr[i] != 0.f)
          check = false;
    }
  }

  BOOST_CHECK_EQUAL(check, true);
}


BOOST_AUTO_TEST_CASE(FilterBlock_through_GPU) {
  FilterBlock fb;
  fb.initialize();
  fb.setMode(T_GPU);
  int num_taps = 512;
  int num_inputs = 2;
  int num_outputs = 2;
  int frame_len = 512; 

  std::vector<float> buffer_in;
  buffer_in.assign(num_inputs*frame_len, 0.f);
  std::vector<float> buffer_out;
  buffer_out.assign(num_outputs*frame_len, 0.f);

  fb.setFrameLen(frame_len);
  fb.setNumInAndOutputs(num_inputs, num_outputs);
  fb.initialize();
  BOOST_CHECK_EQUAL(fb.getFilterContainerSize(), 4*fb.getFilterLen());
  for(int i = 0; i < num_inputs; i++){
    for(int j = 0; j < frame_len; j++){
      buffer_in.at(i*frame_len+j) = (float)(i+1)*j;
    }
  }
  
  clock_t start_t;
  clock_t end_t;
  start_t = clock();  

  fb.frameThrough(&(buffer_in[0]), &(buffer_out[0]));

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"Pass through kernel - time: %f ms") 
                    % ((float)end_t/CLOCKS_PER_SEC*1000.f);
  /*
  for(int i = 0; i < num_outputs; i++){
    for(int j = 0; j < frame_len; j++){
      BOOST_CHECK_EQUAL(buffer_out.at(i*frame_len+j), (float)(i+1)*j);
    }
  }
  */
}

BOOST_AUTO_TEST_CASE(FilterBlock_convolve_GPU) {
  FilterBlock fb;
  fb.initialize();
  fb.setMode(T_GPU);
  int num_taps = 48000;
  int num_inputs = 2;
  int num_outputs = 2;
  int frame_len = 256; 

  std::vector<float> buffer_in;
  buffer_in.assign(num_inputs*frame_len, 0.f);
  std::vector<float> buffer_out;
  buffer_out.assign(num_outputs*frame_len, 0.f);

  std::vector<float> filter1(num_taps, 0.f);
  std::vector<float> filter2(num_taps, 0.f);

  filter1.at(0) = 1.f;
  filter1.at(1) = 1.f;
  filter1.at(num_taps-1) = 1.f;

  fb.setFrameLen(frame_len);
  fb.setFilterLen(num_taps);
  fb.setNumInAndOutputs(num_inputs, num_outputs);
  fb.setFilterTaps(0,0, filter1);
  fb.setFilterTaps(0,1, filter1);
  fb.setFilterTaps(1,0, filter1);
  fb.setFilterTaps(1,1, filter1);
  fb.initialize();

  BOOST_TEST_MESSAGE("Checking filter container size");
  BOOST_CHECK_EQUAL(fb.getFilterContainerSize(), 4*fb.getFilterLen());
  
  // Buffer that goes in is interlaced
  for(int i = 0; i < num_inputs; i++){
    for(int j = 0; j < frame_len; j++){
      buffer_in.at(j*num_inputs+i) = (float)(i+1)*j;
    }
  }
  
  BOOST_TEST_MESSAGE("Launch frame convolution");
  clock_t start_t;
  clock_t end_t;
  start_t = clock();  

  fb.convolveFrameGPU(&(buffer_in[0]), &(buffer_out[0]));
  //fb.convolveFrameGPU(&(buffer_in[0]), &(buffer_out[0]));
  //fb.convolveFrameGPU(&(buffer_in[0]), &(buffer_out[0]));

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"Convolve kernel - time: %f ms") 
                    % ((float)end_t/CLOCKS_PER_SEC*1000.f);
  
  // Output is interlaced too
  BOOST_TEST_MESSAGE("Checking results");
  for(int i = 0; i < num_outputs; i++){
    for(int j = 0; j < frame_len; j++) {
       printf("out %f  in %f\n", buffer_out.at(j*num_inputs+i), buffer_in.at(j*num_inputs+i));
      //BOOST_CHECK_EQUAL(buffer_out.at(j*num_outputs+i), (float)(i+1)*j);
    }
  } 
  BOOST_TEST_MESSAGE("All done"); 
}