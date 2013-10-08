#ifndef SIGNAL_BLOCK_H
#define SINGAL_BLOCK_H


/* SignalBlock class generates and analyzes signals

 Currently the class has functionality to generate 
 and analyze swept sine excitation with fftw library
 for impulse response measurement
 
*/


#include "../global_includes.h"
#include <vector>
#include <cmath>
#include <algorithm>

class SignalBlock {
public:
  SignalBlock()
  : fs_(48000.f),
    length_(3.f),
    f_begin_(50.f),
    f_end_(20e3f),
    padding_begin_(0.1f),
    padding_end_(0.1f)
  {};

  ~SignalBlock() {};

private:
  float fs_;
  float length_;
  float f_begin_;
  float f_end_;
  float padding_begin_;
  float padding_end_;

public:
  float getFs() {return this->fs_;};
  float getLength() {return this->length_;};
  float getFBegin() {return this->f_begin_;};
  float getFEnd() {return this->f_end_;};
  float getPaddingEnd() {return this->padding_begin_;};
  float getPaddingBegin() {return this->padding_end_;};
  void setFs(float fs) {this->fs_=fs;};
  void setLength(float length) {this->length_=length;};
  void setFBegin(float f_begin) {this->f_begin_=f_begin;};
  void setFEnd(float f_end) {this->f_end_=f_end;};
  void setPaddingBegin(float padding) {this->padding_begin_=padding;};
  void setPaddingEnd(float padding) {this->padding_end_=padding;};

  std::vector<float> getSine();
  std::vector<float> getSweep();

  std::vector<float> getRawIr(std::vector<float> sweep_signal,
                              std::vector<float> measured_signal);

};

#endif
