#ifndef CUDAHELPER_H
#define CUDAHELPER_H

#include "../global_includes.h"
#include <cuda.h>
#include <cuda_runtime.h>

// OpenGL Graphics includes
#include <GL/glew.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <stdio.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

extern "C" {
  void c_log_msg(log_level level, const char* msg, ...);
}


// Helper function to check error codes
inline void cudasafe( cudaError_t error, const char* message) {
	const char * errorStr = cudaGetErrorString(error);
	if(error!=cudaSuccess) { c_log_msg(LOG_ERROR,"ERROR: %s : %s\n", message, errorStr); throw(-1);
	}
};


// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId();

int _ConvertSMVer2Cores(int major, int minor); 

#endif
