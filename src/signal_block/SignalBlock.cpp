#include <fftw3.h>

#include "SignalBlock.h"

fftw_complex* multiply(fftw_complex* a, fftw_complex* b, fftw_complex* c) {
  (*c)[0] = (*a)[0]*(*b)[0]-(*a)[1]*(*b)[1];
  (*c)[1] = (*a)[0]*(*b)[1]+(*a)[1]*(*b)[0];
}

fftw_complex* divide(fftw_complex* a, fftw_complex* b, fftw_complex* c) {
  double denum = 1/((*b)[0]*(*b)[0]+(*b)[1]*(*b)[1]); // c^2 + d^2
  double real_num = (*a)[0]*(*b)[0]+(*a)[1]*(*b)[1]; // ac + bd 
  double imag_num = (*a)[1]*(*b)[0]-(*a)[0]*(*b)[1]; // bc - ad
  (*c)[0] = real_num*denum;
  (*c)[1] = imag_num*denum;
}

std::vector<float> SignalBlock::getSine() {
  log_msg<LOG_INFO>(L"SignalBlock::getSine()\n"
                    "  Length: %f, Frequency: %f\n"
                    "  Padding begin: %f end: %f")
                    %this->getLength() %this->getFBegin()
                    %this->getPaddingBegin() %this->getPaddingEnd();

  // tune length so that full it ends at zero
  int cycles = this->getLength()*this->getFBegin();  
  int length = (((float)(cycles+1))*1.f/this->getFBegin())*this->getFs();
  int pad_begin_idx = this->getPaddingBegin()*this->getFs();
  int pad_end_idx = this->getPaddingEnd()*this->getFs();

  log_msg<LOG_DEBUG>(L"SignalBlock::getSine()"
                     "cycles: %d, length: %d, pad_begin: %d, pad_end: %d")
                     %cycles % length %pad_begin_idx %pad_end_idx;

  std::vector<float> ret;
  ret.assign(pad_begin_idx+length+pad_end_idx, 0.f);

  for(int i = 0; i < (pad_begin_idx+length); i++) {
    if(i >= pad_begin_idx)
      ret.at(i) = (sinf(2*M_PI*((i-pad_begin_idx)/this->getFs())*this->getFBegin()));
  }
  return ret;
}

std::vector<float> SignalBlock::getSweep() {
  log_msg<LOG_INFO>(L"SignalBlock::getSweep()\n"
                    "  Length: %f, Begin Frequency: %f, End Frequency: %f \n"
                    "  Padding begin: %f end: %f")
                    %this->getLength() %this->getFBegin() %this->getFEnd()
                    %this->getPaddingBegin() %this->getPaddingEnd();

   int pad_begin_idx = this->getPaddingBegin()*this->getFs();
   int pad_end_idx = this->getPaddingEnd()*this->getFs();
   int length = this->getLength()*this->getFs();
    
   std::vector<float> ret;
   ret.assign(pad_begin_idx+length+pad_end_idx, 0.f);

   for(int i = 0; i < (pad_begin_idx+length); i++) {
    if(i >= pad_begin_idx) {
      int time_idx = i-pad_begin_idx;
      float time_fraction = ((float)time_idx)/((float)length);

      float ratio = log(this->getFEnd()/this->getFBegin());
      float term_1 = (2*M_PI*this->getFBegin()*this->getLength())/ratio;
      float term_2 = exp((time_fraction)*ratio)-1.f;
      ret.at(i) = sinf(term_1*term_2);
    }
  }

  return ret;
}

std::vector<float> SignalBlock::getRawIr(const std::vector<float> sweep,
                                         const std::vector<float> measured) {
  
  clock_t timer;
  timer = clock();

  int len = sweep.size();
  log_msg<LOG_INFO>(L"SignalBlock::getRawIr - Extracting IR with fftw, Sweep length %d") %len;


  std::vector<float> ir;

  fftw_complex* in_sweep = (fftw_complex*)malloc(sizeof(fftw_complex)*len);
  fftw_complex* out_sweep = (fftw_complex*)malloc(sizeof(fftw_complex)*len);
  fftw_complex* in_measured = (fftw_complex*)malloc(sizeof(fftw_complex)*len);
  fftw_complex* out_measured = (fftw_complex*)malloc(sizeof(fftw_complex)*len);
  
  for(int i = 0; i < len; i++) {
    in_sweep[i][0] = (double)sweep.at(i); 
    in_sweep[i][1] = 0;
    in_measured[i][0] = (double)measured.at(i);
    in_measured[i][1] = 0;
  }

  fftw_plan p = fftw_plan_dft_1d(len, in_sweep, out_sweep, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  
  p = fftw_plan_dft_1d(len, in_measured, out_measured, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  for(int i = 0; i < len; i ++) {
    divide(&out_measured[i], &out_sweep[i], &in_sweep[i]);
    //multiply(&out_sweep[i], &out_measured[i], &in_sweep[i]);
  }

  p = fftw_plan_dft_1d(len, in_sweep, out_measured, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  ir.assign(len, 0.f);
  for(int i = 0; i < len; i++)
    ir.at(i) = out_measured[i][0];

  fftw_free(in_sweep);
  fftw_free(out_sweep);
  fftw_free(in_measured);
  fftw_free(out_measured);
  timer = clock()-timer;
  float time = (float)timer/CLOCKS_PER_SEC*1e3;

  log_msg<LOG_INFO>(L"SignalBlock::getRawIr - Done in %f ms") % time;

  return ir;
}

