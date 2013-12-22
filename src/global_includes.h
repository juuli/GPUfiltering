#ifndef PA_GLOBAL_INCLUDES_H
#define PA_GLOBAL_INCLUDES_H

#include "logger.h"


enum EXEC_MODE {T_CPU, T_GPU, FFT_CPU, FFT_GPU, SWEEP};

static const char* mode_texts[] = {"Time Domain CPU",
                                   "Time Domain GPU",
                                   "FFT CPU via FFTW",
                                   "FFT GPU via cuFFT",
                                   "Log Sweep Measurement"};

#endif
