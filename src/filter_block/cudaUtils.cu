#include "cudaUtils.h"


int getCurrentDevice() {
  int dev_num;
  cudasafe(cudaGetDevice(&dev_num), "cudaGetDevice");
  return dev_num;
}

void printMemInfo(char* message) {
  size_t total_mem = 0;
  size_t free_mem = 0;
  size_t total_mem_1 = 0;
  size_t free_mem_1 = 0;
  int current_device = getCurrentDevice();

  cudaSetDevice(0);
  cudasafe(cudaMemGetInfo (&free_mem, &total_mem), "Cuda meminfo");
  cudaSetDevice(1);
  cudasafe(cudaMemGetInfo (&free_mem_1, &total_mem_1), "Cuda meminfo");
  cudaDeviceSynchronize();

  c_log_msg(LOG_DEBUG, "%s - device %u, mem_size %u MB, free %u MB",
                        message, 0, total_mem/1000000, free_mem/1000000);

  c_log_msg(LOG_DEBUG, "%s - device %u, mem_size %u MB, free %u MB",
                        message, 1, total_mem_1/1000000, free_mem_1/1000000);

  cudaSetDevice(current_device);
}

/////////////////////
// Checkkers

float printCheckSum(float* d_data, size_t mem_size, char* message) {
  float sum = 0.f;
  if(IGNORE_CHECKSUMS == 0) {
    float* h_data = (float*)calloc(mem_size, sizeof(float));
    sum = 0.f;

    cudasafe(cudaMemcpy(h_data, d_data, mem_size*sizeof(float), cudaMemcpyDeviceToHost), "Memcopy");

    for(size_t i = 0; i < mem_size; i++)
      sum += h_data[i];

    c_log_msg(LOG_INFO, "kernels3d.cu: printChecksum float - %s checksum: %f", message, sum);
    free(h_data);
  }

  return sum;
}

void printCheckSum(unsigned char* d_data, size_t mem_size, char* message) {
  if(IGNORE_CHECKSUMS == 0) {
    unsigned char* h_data = (unsigned char*)calloc(mem_size, sizeof(unsigned char));
    unsigned int sum = 0;

    cudasafe(cudaMemcpy(h_data, d_data, mem_size*sizeof(unsigned char), cudaMemcpyDeviceToHost), "Memcopy");

    for(size_t i = 0; i < mem_size; i++)
      sum += (unsigned int)h_data[i];

    c_log_msg(LOG_INFO, "kernels3d.cu: printChecksum unsigned char - %s checksum: %u", message, sum);

    free(h_data);
  }
}

void printMax(float* d_data, size_t mem_size, char* message) {
  if(IGNORE_CHECKSUMS == 0) {
    float* h_data = (float*)calloc(mem_size, sizeof(float));
     float max_val = -999999999999.f;

    cudasafe(cudaMemcpy(h_data, d_data, mem_size*sizeof(float), cudaMemcpyDeviceToHost), "Memcopy");

    for(size_t i = 0; i < mem_size; i++){
      if(h_data[i] > max_val)
        max_val = h_data[i]; 
    }

    c_log_msg(LOG_INFO, "kernels3d.cu: printMax - %s Maximum value: %f", message, max_val);

    free(h_data);
  }
}

/// Kernel

template <>
void resetData(unsigned int mem_size, float* d_data, unsigned int device) {
  c_log_msg(LOG_DEBUG, "cudaUtils.cu: resetFloats(data) - mem_size %u, device %d", mem_size, device);
  cudasafe(cudaSetDevice(device), "floatsToDevice: cudaSetDevice");

  dim3 block(128);
  dim3 grid(mem_size/block.x+1);
  resetKernel< float ><<<grid, block>>>(d_data, mem_size);
}

template <>
void resetData(unsigned int mem_size, double* d_data, unsigned int device) {
  c_log_msg(LOG_DEBUG, "cudaUtils.cu: resetDouble(data) - mem_size %u, device %d", mem_size, device);

  dim3 block(128);
  dim3 grid(mem_size/block.x+1);
  resetKernel< double ><<<grid, block>>>(d_data, mem_size);
}

template <typename T>
__global__ void resetKernel(T* d_data, unsigned int mem_size) {
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  if(idx < mem_size)
    d_data[idx] = (T)0;
}
