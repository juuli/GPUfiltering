#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <ctime>
#include <boost/test/unit_test.hpp>
#include "../filter_block/FilterBlock.h"
#include "../filter_block/GPUfilter.h"

#define DEVICE 0
#define NUM_ITERATIONS 20

void multiply(fftw_complex* a, fftw_complex* b, fftw_complex* c) {
  (*c)[0] = (*a)[0]*(*b)[0]-(*a)[1]*(*b)[1];
  (*c)[1] = (*a)[0]*(*b)[1]+(*a)[1]*(*b)[0];
}

void multiplyAdd(fftw_complex* a, fftw_complex* b, fftw_complex* c) {
  (*c)[0] += (*a)[0]*(*b)[0]-(*a)[1]*(*b)[1];
  (*c)[1] += (*a)[0]*(*b)[1]+(*a)[1]*(*b)[0];
}

BOOST_AUTO_TEST_CASE(Check_cuda_utils) {
  FilterBlock fb;
  
  unsigned int device = 0;
  cudaSetDevice(device);
  cudaDeviceReset();

  unsigned int filter_len = FILTER_LEN;
  unsigned int buff_size = BUFFER_LEN;
 
  float* d_inbuffer = valueToDevice<float>(buff_size, 0.f, device);
  float* d_outbuffer = valueToDevice<float>(buff_size, 0.f, device);
  float* d_filter = valueToDevice<float>(filter_len, 0.f, device);
  float* d_conv_buf = valueToDevice<float>((filter_len+buff_size-1), 0.f, device);

  std::vector<float> cont;
  cont.assign(10, 0.f);
  cont.at(0) = 1.f;
  cont.at(1) = 1.f;
  copyHostToDevice<float>(2, d_inbuffer, &(cont[0]), device);
  copyHostToDevice<float>(2, d_inbuffer+100, &(cont[0]), device);

  // Remember to Free these!
  float* h_ret = fromDevice<float>(200, d_inbuffer, device);

  // Check that the copy works
  for(int i = 0; i < 10; i++) {
    BOOST_CHECK_EQUAL(h_ret[i], cont.at(i));  
    BOOST_CHECK_EQUAL(h_ret[i+100], cont.at(i));  

  }
  
  // Reset function
  resetData(buff_size, d_inbuffer, 0);
  free(h_ret);
  h_ret = (float*)NULL;
  h_ret = fromDevice<float>(buff_size, d_inbuffer, device);
  for(int i = 0; i < 10; i++)
    BOOST_CHECK_EQUAL(h_ret[i], 0.f);  
  


  destroyMem<float>(d_inbuffer);
  destroyMem<float>(d_outbuffer);
  destroyMem<float>(d_filter);
  destroyMem<float>(d_conv_buf);
  free(h_ret);
  cudaDeviceReset();
}

BOOST_AUTO_TEST_CASE(CPU_convolve) {
  unsigned int device = 0;
  cudaSetDevice(device);
  cudaDeviceReset();

  unsigned int filter_len = FILTER_LEN;
  unsigned int buffer_len = BUFFER_LEN;
  unsigned int conv_len = filter_len+buffer_len-1;

  std::vector<float> buffer(buffer_len,  0.f);
  std::vector<float> filter(filter_len,  0.f);
  std::vector<float> result(conv_len, 0.f);

  buffer.at(0) = 1.f;
  buffer.at(1) = 1.f;
  filter.at(1) = 1.f;
  filter.at(2) = 1.f;
  filter.at(10) = 1.f;
  filter.at(filter_len-1) = 1.f;

  clock_t start_t;
	clock_t end_t;

  ///////////////////
  // Benchmark loop

  float times = 0.f;
  for(int i = 0; i < NUM_ITERATIONS; i++) {

    start_t = clock();  
    convolutionCPU(&(buffer[0]), &(filter[0]), &(result[0]),
                    buffer_len, filter_len);

  
    times += (float)(clock()-start_t);
  } // end iteration loop
  times /= NUM_ITERATIONS;

  end_t = clock()-start_t;
  log_msg<LOG_INFO>(L"Convolution CPU - time: %f ms") 
              % (times/CLOCKS_PER_SEC*1000);

  // Vector to check the right values
  std::vector<float> values(conv_len, 0.f);
  values.at(1) = 1.f;
  values.at(2) = 2.f;
  values.at(3) = 1.f;
  values.at(10) = 1.f;
  values.at(11) = 1.f;
  values.at(filter_len-1) = 1.f;
  values.at(filter_len) = 1.f;

  for(int i = 0; i < (buffer_len+filter_len-1); i++){
    BOOST_CHECK_EQUAL(result.at(i), values.at(i));
    /*
    if(i == 2){
      BOOST_CHECK_EQUAL(result.at(i), 2.f);
    }
    else if(i == 1 || i == 3 || i == 10 || i == 11){
      BOOST_CHECK_EQUAL(result.at(i), 1.f);
    }
    else
      BOOST_CHECK_EQUAL(result.at(i), 0.f);
    */
  }
}

BOOST_AUTO_TEST_CASE(GPU_convolve) {
  unsigned int device = DEVICE;
  cudaSetDevice(device);
  cudaDeviceReset();

  unsigned int filter_len = FILTER_LEN;
  unsigned int buffer_len = BUFFER_LEN;
  unsigned int conv_len = filter_len+buffer_len-1;

  std::vector<float> buffer(buffer_len,  0.f);
  std::vector<float> filter(filter_len,  0.f);
  std::vector<float> result(conv_len, 0.f);

  buffer.at(0) = 1.f;
  buffer.at(1) = 1.f;
  filter.at(1) = 1.f;
  filter.at(2) = 1.f;
  filter.at(10) = 1.f;
  filter.at(filter_len-1) = 1.f;

  clock_t start_t;
	clock_t end_t;


  float* d_x = valueToDevice<float>(buffer_len, 0.f, 0);
  float* d_h = toDevice<float>(filter_len, &(filter[0]), 0);
  float* d_y = valueToDevice<float>(conv_len, 0.f, 0);

  ///////////////////
  // Benchmark loop

  float times = 0.f;
  for(int i = 0; i < NUM_ITERATIONS; i++) {

    start_t = clock();  

    copyHostToDevice<float>(buffer_len, d_x, &(buffer[0]), device);

    convolutionGPU(d_x, d_h, d_y, buffer_len, filter_len);

    copyDeviceToHost<float>(conv_len, &(result[0]), d_y, device);

    times += (float)(clock()-start_t);

  } // end iteration loop
  
  times /= NUM_ITERATIONS;

  end_t = clock()-start_t;
	log_msg<LOG_INFO>(L"Convolution GPU - time: %f ms") 
					          % ((float)times/CLOCKS_PER_SEC*1000.f);

  destroyMem(d_x);
  destroyMem(d_h);
  destroyMem(d_y);

  // Vector to check the right values
  std::vector<float> values(conv_len, 0.f);
  values.at(1) = 1.f;
  values.at(2) = 2.f;
  values.at(3) = 1.f;
  values.at(10) = 1.f;
  values.at(11) = 1.f;
  values.at(filter_len-1) = 1.f;
  values.at(filter_len) = 1.f;

  for(int i = 0; i < (buffer_len+filter_len-1); i++){
    BOOST_CHECK_MESSAGE(result.at(i)== values.at(i), 
                        "not matchig "<<result.at(i)<<"!="<<values.at(i)<< " at" << i);
  }
}

/*
BOOST_AUTO_TEST_CASE(GPU_convolve_Padded) {
  unsigned int device = DEVICE;
  cudaSetDevice(device);
  cudaDeviceReset();

  unsigned int filter_len = FILTER_LEN;
  unsigned int buffer_len = BUFFER_LEN;
  unsigned int conv_len = filter_len+buffer_len-1;

  std::vector<float> buffer(buffer_len,  0.f);
  std::vector<float> filter(filter_len,  0.f);
  std::vector<float> result(conv_len, 0.f);

  buffer.at(0) = 1.f;
  buffer.at(1) = 1.f;
  filter.at(1) = 1.f;
  filter.at(2) = 1.f;
  filter.at(10) = 1.f;
  filter.at(filter_len-1) = 1.f;

  clock_t start_t;
	clock_t end_t;

  int pad = buffer_len-1;
  float* d_x = valueToDevice<float>(buffer_len, 0.f, 0);
  float* d_h = valueToDevice<float>(filter_len+(pad)*2, 0.f, 0);
  float* d_y = valueToDevice<float>(conv_len, 0.f, 0);

  // this loop should be clock "real-time" values
  start_t = clock();  
  copyHostToDevice<float>(filter_len, d_h+pad, &(filter[0]), device);
  copyHostToDevice<float>(buffer_len, d_x, &(buffer[0]), device);

  convolutionGPUpadH(d_x, d_h, d_y, buffer_len, filter_len);

  copyDeviceToHost<float>(conv_len, &(result[0]), d_y, device);

  end_t = clock()-start_t;
	log_msg<LOG_INFO>(L"Convolution GPU PAD - time: %f ms") 
					          % ((float)end_t/CLOCKS_PER_SEC*1000.f);

  destroyMem(d_x);
  destroyMem(d_h);
  destroyMem(d_y);

  // Vector to check the right values
  std::vector<float> values(conv_len, 0.f);
  values.at(1) = 1.f;
  values.at(2) = 2.f;
  values.at(3) = 1.f;
  values.at(10) = 1.f;
  values.at(11) = 1.f;
  values.at(filter_len-1) = 1.f;
  values.at(filter_len) = 1.f;

  for(int i = 0; i < (buffer_len+filter_len-1); i++){
    BOOST_CHECK_MESSAGE(result.at(i)== values.at(i), 
                        "not matchig "<<result.at(i)<<"!="<<values.at(i)<< " at " << i);
  }
  cudaDeviceReset();
}
*/
BOOST_AUTO_TEST_CASE(GPU_convolve_Shared) {
  unsigned int device = DEVICE;
  cudaSetDevice(device);
  cudaDeviceReset();

  unsigned int filter_len = FILTER_LEN;
  unsigned int buffer_len = BUFFER_LEN;
  unsigned int conv_len = filter_len+buffer_len-1;

  std::vector<float> buffer(buffer_len,  0.f);
  std::vector<float> filter(filter_len,  0.f);
  std::vector<float> result(conv_len, 0.f);

  buffer.at(0) = 1.f;
  buffer.at(1) = 1.f;
  filter.at(1) = 1.f;
  filter.at(2) = 1.f;
  filter.at(10) = 1.f;
  filter.at(filter_len-1) = 1.f;

  clock_t start_t;
	clock_t end_t;

  int pad = buffer_len-1;
  float* d_x = valueToDevice<float>(buffer_len, 0.f, 0);
  float* d_h = valueToDevice<float>(filter_len+(pad)*2, 0.f, 0);
  float* d_y = valueToDevice<float>(conv_len, 0.f, 0);


  copyHostToDevice<float>(filter_len, d_h+pad, &(filter[0]), device);
  
  // this loop should be clock "real-time" values  
  float times = 0.f;
  for(int i = 0; i < NUM_ITERATIONS; i++) {

    start_t = clock(); 
    copyHostToDevice<float>(buffer_len, d_x, &(buffer[0]), device);
    convolutionGPUshared(d_x, d_h, d_y, buffer_len, filter_len);
    copyDeviceToHost<float>(conv_len, &(result[0]), d_y, device);

    times += (float)(clock()-start_t);

  } // end iteration loop

  times /= NUM_ITERATIONS;
	log_msg<LOG_INFO>(L"Convolution GPU Shared - time: %f ms") 
					          % ((float)times/CLOCKS_PER_SEC*1000.f);



  destroyMem(d_x);
  destroyMem(d_h);
  destroyMem(d_y);

  // Vector to check the right values
  std::vector<float> values(conv_len, 0.f);
  values.at(1) = 1.f;
  values.at(2) = 2.f;
  values.at(3) = 1.f;
  values.at(10) = 1.f;
  values.at(11) = 1.f;
  values.at(filter_len-1) = 1.f;
  values.at(filter_len) = 1.f;

  for(int i = 0; i < (buffer_len+filter_len-1); i++){
    BOOST_CHECK_MESSAGE(result.at(i)== values.at(i), 
                        "not matchig "<<result.at(i)<<"!="<<values.at(i)<< " at " << i);
  }
  cudaDeviceReset();
}


BOOST_AUTO_TEST_CASE(CPU_convolve_FFT) {
  int filter_len_o = FILTER_LEN;
  int buffer_len = BUFFER_LEN;
  
  // Size of the filter needs to be a multiple of the buffer size
  int filter_len = INC_TO_MOD(filter_len_o, buffer_len);
  int conv_len = filter_len+buffer_len-1;
  int pad = buffer_len-1;
  
  std::vector<float> buffer(buffer_len,  0.f);
  std::vector<float> filter(filter_len,  0.f);
  std::vector<float> result;
   
  buffer.at(0) = 1.f;
  buffer.at(1) = 1.f;

  filter.at(1) = 1.f;
  filter.at(2) = 1.f;
  filter.at(10) = 1.f;
  filter.at(filter_len_o-1) = 1.f;

  // L is the length of the transform
  int L = buffer_len+pad;

  // well be missing one here , lets think about it later
  int num_filter_parts = filter_len/buffer_len;  
  result.assign((num_filter_parts+2)*buffer_len, 0.f);
   
  // Padded input buffer
  fftw_complex* x_empty = (fftw_complex*)malloc(sizeof(fftw_complex)*L);
  fftw_complex* x = (fftw_complex*)malloc(sizeof(fftw_complex)*L);

  // Fragmented filter, frequency domain
  fftw_complex** H = (fftw_complex**)malloc(sizeof(fftw_complex*)*num_filter_parts);
  
  // Frequency delay line
  fftw_complex** fdl = (fftw_complex**)malloc(sizeof(fftw_complex*)*num_filter_parts);

  // Input/output data  
  fftw_complex* fft_in = (fftw_complex*)malloc(sizeof(fftw_complex)*L);
  fftw_complex* fft_out = (fftw_complex*)malloc(sizeof(fftw_complex)*L);
  fftw_plan fft_p = fftw_plan_dft_1d(L, fft_in, fft_out, 
                                     FFTW_FORWARD, FFTW_ESTIMATE);

  fftw_plan ifft_p = fftw_plan_dft_1d(L, fft_in, fft_out, 
                                      FFTW_BACKWARD, FFTW_ESTIMATE);

  // Initialize accummulation buffer
  fftw_complex* accumulator = (fftw_complex*)malloc(sizeof(fftw_complex)*L);

  // Initialize input buffers (one full, one blank) and the accummulator
  for(int i = 0; i < L; i++) {
    x_empty[i][0] = 0;
    x_empty[i][1] = 0;
    x[i][0] = 0;
    x[i][1] = 0;
    accumulator[i][0] = 0;
    accumulator[i][1] = 0;
    fft_in[i][0] = 0;
    fft_in[i][1] = 0;
    fft_out[i][0] = 0;
    fft_out[i][0] = 0;

    if(i >= pad) {
      x[i][0] = buffer.at(i-pad);
      //printf("buffer at %i, %f\n", i, x[i][0]);
    }
  }

  // Assign and copy filter fractions to memory
  for(int i = 0; i < num_filter_parts; i++) {
    H[i] = (fftw_complex*)malloc(sizeof(fftw_complex)*L);
    fdl[i] = (fftw_complex*)malloc(sizeof(fftw_complex)*L);

    for(int j = 0; j < L; j++) {
      // Reset the frequency delay line
      fdl[i][j][0] = 0;
      fdl[i][j][1] = 0;
      H[i][j][0] = 0;
      H[i][j][1] = 0;

      ///// Initialize the filters
      // f_idx : index from the whole filter
      int f_idx = i*buffer_len+j;

      if(f_idx < filter_len) {
        // grab the buffer_len first taps
        fft_in[j][0] = (j<buffer_len) ? filter.at(f_idx) : 0;
        fft_in[j][1] = 0;
      }      
    }
    // Run fft
    fftw_execute(fft_p);
    memcpy((void*)&(H[i][0]), (void*)fft_out, sizeof(fftw_complex)*L);
  }  

  clock_t start_t;
	clock_t end_t;

  ///////////////////
  // Benchmark loop

  float times = 0;
    for(int i = 0; i < NUM_ITERATIONS; i++) {

    start_t = clock();  

    /////////////////////
    // Actual convolution
    //
    // 1: fft the next input buffer
    // 2: append the delay line with the spectrum of the new buffer
    // 3: multiply the dealy line spectrums with filter spectrums
    // 4: accummulate
    // 5: copy output buffer

    for(int i = 0; i < num_filter_parts+2; i++) {
      // calculate the buffer index
      int b_idx = MOD(i, num_filter_parts);

      ///////////
      //1
      if(i == 0) // if 0, get the "full" buffer
        memcpy((void*)fft_in, (void*)x, sizeof(fftw_complex)*L);
      else
        memcpy((void*)fft_in, (void*)x_empty, sizeof(fftw_complex)*L);

      fftw_execute(fft_p);

      /////////////
      //2
      memcpy((void*)&(fdl[b_idx][0]), fft_out, sizeof(fftw_complex)*L);    

      ////////////
      //3&4 Muliply and accumulate the spectrums
      for(int j = 0; j < num_filter_parts; j++){
        // index of the previous buffer
        int buf_idx = MOD(b_idx-j, num_filter_parts);

        // Multiply and accumulate
        for(int k = 0; k < L; k++){
          multiplyAdd(&fdl[buf_idx][k], &H[j][k], &(accumulator[k]));
        } // end k loop
      } // end j loop
          
      ////////////
      //5
      memcpy((void*)fft_in, (void*)accumulator, sizeof(fftw_complex)*L);
      fftw_execute(ifft_p);
      
      for(int j = 0; j < L; j++) {
        if(j < buffer_len){
          result.at(i*buffer_len+j) = fft_out[j+pad][0]/L;
        }
        accumulator[j][0] = 0;
        accumulator[j][1] = 0;
      } // end j loop
    } // end i loop

    end_t = clock()-start_t;
  	
    times += (float)(clock()-start_t);


    } // end iteration loop
  times /= NUM_ITERATIONS;

  log_msg<LOG_INFO>(L"Convolution CPU fft - time: %f ms") 
					          % (times/CLOCKS_PER_SEC*1000.f);

  // Vector to check the right values
  std::vector<float> values(conv_len, 0.f);
  values.at(1) = 1.f;
  values.at(2) = 2.f;
  values.at(3) = 1.f;
  values.at(10) = 1.f;
  values.at(11) = 1.f;
  values.at(filter_len_o-1) = 1.f;
  values.at(filter_len_o) = 1.f;
 
  //std::cout<<std::numeric_limits<double>::epsilon()<<" \n" << std::scientific;
  
  for(int i = 0; i < conv_len; i++) {
    float val = (result.at(i)>std::numeric_limits<double>::epsilon()) ? result.at(i) : 0;
    BOOST_CHECK_MESSAGE(val == values.at(i), 
                        "not matchig "<<result.at(i)<<"!="<<values.at(i)<< " at " << i);
  }
 
  // cleanup
  fftw_destroy_plan(fft_p);
  fftw_destroy_plan(ifft_p);
  free(x);
  free(x_empty);
  free(fft_in);
  free(fft_out);
  free(accumulator);

  for(int i = 0; i < num_filter_parts; i++) {
    free(H[i]);
    free(fdl[i]);
  }
  free(fdl);
  free(H); 
}

BOOST_AUTO_TEST_CASE(GPU_convolve_FFT) {
  unsigned int filter_len_o = FILTER_LEN;
  unsigned int buffer_len = BUFFER_LEN;
  
  /////////
  // Size of the filter needs to be a multiple of the buffer size
  // to have "full" partitions

  int filter_len = INC_TO_MOD(filter_len_o, buffer_len);
  int conv_len = filter_len+buffer_len-1;
  int pad = buffer_len-1;
  
  std::vector<float> buffer(buffer_len,  0.f);
  std::vector<float> filter(filter_len,  0.f);
   
  buffer.at(0) = 1.f;
  buffer.at(1) = 1.f;
  filter.at(1) = 1.f;
  filter.at(2) = 1.f;
  filter.at(10) = 1.f;
  filter.at(filter_len_o-1) = 1.f;

  // L is the length of the transform
  int L = buffer_len+pad;
  int num_filter_parts = filter_len/buffer_len;  

  // Have an extra parts to accommodate the result
  std::vector<float> result(buffer_len*(num_filter_parts+2), 0.f);

  // Padded input buffer
  cufftComplex val;
  val.x = 0.f; val.y = 0.f;

  std::vector<cufftComplex> buf(L);
  std::vector<cufftReal> buf_r(L);
  buf.assign(L, val);
  buf_r.assign(L, 0.f);

  for(int i = 0; i < L; i ++)
    if(i >= pad) {
      buf.at(i) = make_float2(buffer.at(i-pad), 0.f);
      buf_r.at(i) = buffer.at(i-pad);
    }
    else {
      buf.at(i) = make_float2(0.f, 0.f);
      buf_r.at(i) = 0.f;
    }

  // Input buffers
  cufftComplex* x_empty = valueToDevice<cufftComplex>(L, val, 0);
  cufftComplex* x = toDevice<cufftComplex>(L, &(buf[0]), 0);
  // REAL
  cufftReal* x_empty_r = valueToDevice<cufftReal>(L, 0.f, 0);
  cufftReal* x_r = toDevice<cufftReal>(L, &(buf_r[0]), 0);

  // Frequency delay line size: L x number of filter parts
  cufftComplex* fdl = valueToDevice<cufftComplex>(num_filter_parts*L, val, 0);

  // Input/output data size: L
  cufftComplex* fft_in = valueToDevice<cufftComplex>(L, val, 0);
  cufftComplex* fft_out = valueToDevice<cufftComplex>(L, val, 0);

  cufftReal* fft_in_r = valueToDevice<cufftReal>(L, 0.f, 0);
  cufftReal* fft_out_r = valueToDevice<cufftReal>(L, 0.f, 0);

  // cufft Plans
  cufftHandle fft_p;
  cufftHandle ifft_p;
  cufftHandle fft_p_N;
  cufftPlan1d(&ifft_p, L, CUFFT_C2R, 1);
  cufftPlan1d(&fft_p, L, CUFFT_R2C, 1);

  // Batch of transforms
  cufftPlan1d(&fft_p_N, L, CUFFT_C2C, num_filter_parts);

  // Initialize accummulation buffer size: L
  cufftComplex* accumulator = valueToDevice<cufftComplex>(L, val, 0);
  std::vector<cufftReal> ret_buf(buffer_len);

  // Initialize filter partitions first at host
  std::vector<cufftComplex> h_host;
  std::vector<cufftReal> h_host_r;
  h_host.assign(num_filter_parts*L, val);
  h_host_r.assign(num_filter_parts*L, 0.f);

  // Assign and copy filter fractions to memory
  for(int i = 0; i < num_filter_parts; i++) {
    for(int j = 0; j < L; j++) {
      ///// Initialize the filters
      int f_idx = i*L+j;
      int h_idx = i*buffer_len+j;
      if(j < buffer_len) {
        h_host.at(f_idx) = make_float2(filter.at(h_idx), 0.f);
        h_host_r.at(f_idx) = filter.at(h_idx);
        //if(h_idx == filter_len_o-1)
        //  printf("tap %f , partition %u / %u \n", filter.at(h_idx), i, num_filter_parts);
      }
    }
  }

  // Fragmented filter, frequency domain
  cufftComplex* H = toDevice<cufftComplex>(num_filter_parts*L, &(h_host[0]),0);
  cudaDeviceSynchronize();

  cufftExecC2C(fft_p_N, H, H, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  
  clock_t start_t;
	clock_t end_t;
  
  ///////////////////
  // Benchmark loop

  float times;
  for(int i = 0; i < NUM_ITERATIONS; i++) {

    start_t = clock();  

    // Actual convolution
    //
    // 1 fft the next input buffer
    // 2 append the delay line with the spectrum of the new buffer
    // 3 multiply the dealy line spectrums with filter spectrums
    // 4 accummulate
    // 5 copy output buffer

    for(int i = 0; i < (num_filter_parts+2); i++) {
      /// Current filter part
      int H_idx = MOD(i, num_filter_parts);

      ///////////
      // 1 
      if(i == 0) // if 0, get the "full" buffer
        copyDeviceToDevice<cufftReal>(L, fft_in_r, x_r, 0);
      else
        copyDeviceToDevice<cufftReal>(L, fft_in_r, x_empty_r, 0);

      // Single fft, buffer to fdl
      cufftExecR2C(fft_p, fft_in_r, fft_out);
      cudaDeviceSynchronize();
    
      /////////////
      // 2
      copyDeviceToDevice<cufftComplex>(L, &(fdl[H_idx*L]), fft_out, 0);
     
      ////////////
      // 3&4 Muliply and accumulate the spectrums
      for(int j = 0; j < num_filter_parts; j++) {
        //Multiply and accumulate
        int fdl_idx = MOD(H_idx-j, num_filter_parts);
        multiplyAddBuffers(fdl, H, accumulator,
                           fdl_idx, H_idx, L);
      } // end j loop
          
      ////////////
      // 5
      cufftExecC2R(ifft_p, accumulator, fft_out_r);
      copyDeviceToHost<cufftReal>(buffer_len, &(ret_buf[0]), (fft_out_r+pad), 0);
      
      for(int j = 0; j < buffer_len; j++) {
        result.at(i*buffer_len+j) = (float)ret_buf[j]/(float)L;
      } // end j loop

      resetData(L, accumulator, 0);
    } // end i loop

    times += (float)(clock()-start_t);


  } // end iteration loop
  times /= NUM_ITERATIONS;

	log_msg<LOG_INFO>(L"Convolution GPU fft - time: %f ms") 
					          %(times/CLOCKS_PER_SEC*1000.f);

  // Vector to check the right values
  std::vector<float> values(buffer_len*(num_filter_parts+2), 0.f);

  values.at(1) = 1.f;
  values.at(2) = 2.f;
  values.at(3) = 1.f;
  values.at(10) = 1.f;
  values.at(11) = 1.f;
  values.at(filter_len_o-1) = 1.f;
  values.at(filter_len_o) = 1.f;
 
  printf("Check \n");
  std::cout<<"Epsilon "<<std::numeric_limits<float>::epsilon()<<" \n" << std::scientific;
  for(int i = 0; i < conv_len; i++){
    //float val = (result.at(i)>std::numeric_limits<float>::epsilon()) ? result.at(i) : 0.f;
    //float val = (result.at(i)>1e-5) ? result.at(i) : 0.f;
    //BOOST_CHECK_MESSAGE(val == values.at(i), 
     //                   "not matchig "<<result.at(i)<<"!="<<values.at(i)<< " at " << i);
  }

  // cleanup
  cufftDestroy(fft_p);
  cufftDestroy(fft_p_N);
  destroyMem(x);
  destroyMem(x_empty);
  destroyMem(x_r);
  destroyMem(x_empty_r);
  destroyMem(fdl);
  destroyMem(H);
  destroyMem(fft_in);
  destroyMem(fft_out);
  destroyMem(fft_in_r);
  destroyMem(fft_out_r);

  destroyMem(accumulator);
}
/*
BOOST_AUTO_TEST_CASE(Assign_filters) {
  int filter_len = 1024;
  int buffer_len = 512;
  
  int pad = buffer_len-1;
  int d_f_len = filter_len+pad*2;
  Convolver c(2, 2, filter_len, buffer_len);
  std::vector< std::vector<float> >filters(2*2);

  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
      std::vector<float> filt(filter_len);
      for(int k = 0; k < filter_len; k++)
        filt.at(k) = k;
        
      filters.at(i*2+j) = filt;
    }
  }
  c.initialize(filters);
  float* d_filters = c.getDFilters();
  float * h_filters = fromDevice<float>(d_f_len*2*2, d_filters, 0);

  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < filter_len; j++) {
      BOOST_CHECK_EQUAL(filters.at(i).at(j), h_filters[i*(filter_len+2*pad)+j+pad]);
    }
  }
  c.cleanup();
  free(h_filters);
}
*/