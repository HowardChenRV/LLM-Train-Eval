/*-----------------------------------------------------------------------*/
/* Program: memcpy_bandwidth                                             */
/* Original code: https://github.com/NVIDIA/cuda-samples                 */
/*                                                                       */
/* This program measures GPU memory transfer rates in GB/s for simple.   */
/*-----------------------------------------------------------------------*/

#include <cuda_runtime.h>
#include <helper_cuda.h>  // helper functions for CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <cuda.h>
#include <cassert>
#include <iostream>
#include <memory>


// defines, project
#define CACHE_CLEAR_SIZE (128 * (1e6))  // 128 M

enum memoryMode { PINNED, PAGEABLE };

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

// if true, use CPU based timing for everything
static bool bDontUseGPUTiming;


extern "C" float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode,
    bool wc, unsigned int nWarmups, unsigned int nRepeats);
extern "C" float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode,
    bool wc, unsigned int nWarmups, unsigned int nRepeats);
extern "C" float testDeviceToDeviceTransfer(unsigned int memSize, 
    unsigned int nWarmups, unsigned int nRepeats);


///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
extern "C" float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode,
                                          bool wc, unsigned int nWarmups, unsigned int nRepeats) {
  StopWatchInterface *timer = NULL;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInGBs = 0.0f;
  unsigned char *h_idata = NULL;
  unsigned char *h_odata = NULL;
  cudaEvent_t start, stop;

  sdkCreateTimer(&timer);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // allocate host memory
  if (PINNED == memMode) {
  // pinned memory mode - use special function to get OS-pinned memory
#if CUDART_VERSION >= 2020
    checkCudaErrors(cudaHostAlloc((void **)&h_idata, memSize,
                                  (wc) ? cudaHostAllocWriteCombined : 0));
    checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize,
                                  (wc) ? cudaHostAllocWriteCombined : 0));
#else
    checkCudaErrors(cudaMallocHost((void **)&h_idata, memSize));
    checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
  } else {
    // pageable memory mode - use malloc
    h_idata = (unsigned char *)malloc(memSize);
    h_odata = (unsigned char *)malloc(memSize);

    if (h_idata == 0 || h_odata == 0) {
      fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
      exit(EXIT_FAILURE);
    }
  }

  // initialize the memory
  for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
    h_idata[i] = (unsigned char)(i & 0xff);
  }

  // allocate device memory
  unsigned char *d_idata;
  checkCudaErrors(cudaMalloc((void **)&d_idata, memSize));

  // initialize memory
  checkCudaErrors(
    cudaMemcpy(d_idata, h_idata, memSize, cudaMemcpyHostToDevice));
  
  // warmup
  for (int r = 0; r < nWarmups; r++) {
    checkCudaErrors(
      cudaMemcpy(h_odata, d_idata, memSize, cudaMemcpyDeviceToHost));
  }

  // copy data from GPU to Host
  if (PINNED == memMode) {
    if (bDontUseGPUTiming) sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < nRepeats; i++) {
      checkCudaErrors(cudaMemcpyAsync(h_odata, d_idata, memSize,
                                      cudaMemcpyDeviceToHost, 0));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
    if (bDontUseGPUTiming) {
      sdkStopTimer(&timer);
      elapsedTimeInMs = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
    }
  } else {
    elapsedTimeInMs = 0;
    for (unsigned int i = 0; i < nRepeats; i++) {
      sdkStartTimer(&timer);
      checkCudaErrors(
          cudaMemcpy(h_odata, d_idata, memSize, cudaMemcpyDeviceToHost));
      sdkStopTimer(&timer);
      elapsedTimeInMs += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      memset(flush_buf, i, FLUSH_SIZE);
    }
  }

  // calculate bandwidth in GB/s
  double time_s = elapsedTimeInMs / 1e3;
  bandwidthInGBs = (memSize * (float)nRepeats) / (double)1e9;
  bandwidthInGBs = bandwidthInGBs / time_s;
  // clean up memory
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaEventDestroy(start));
  sdkDeleteTimer(&timer);

  if (PINNED == memMode) {
    checkCudaErrors(cudaFreeHost(h_idata));
    checkCudaErrors(cudaFreeHost(h_odata));
  } else {
    free(h_idata);
    free(h_odata);
  }

  checkCudaErrors(cudaFree(d_idata));

  return bandwidthInGBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a host to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
extern "C" float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode,
                                          bool wc, unsigned int nWarmups, unsigned int nRepeats) {
  StopWatchInterface *timer = NULL;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInGBs = 0.0f;
  cudaEvent_t start, stop;
  sdkCreateTimer(&timer);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // allocate host memory
  unsigned char *h_odata = NULL;

  if (PINNED == memMode) {
#if CUDART_VERSION >= 2020
    // pinned memory mode - use special function to get OS-pinned memory
    checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize,
                                  (wc) ? cudaHostAllocWriteCombined : 0));
#else
    // pinned memory mode - use special function to get OS-pinned memory
    checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
  } else {
    // pageable memory mode - use malloc
    h_odata = (unsigned char *)malloc(memSize);

    if (h_odata == 0) {
      fprintf(stderr, "Not enough memory available on host to run test!\n");
      exit(EXIT_FAILURE);
    }
  }

  unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
  unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

  if (h_cacheClear1 == 0 || h_cacheClear2 == 0) {
    fprintf(stderr, "Not enough memory available on host to run test!\n");
    exit(EXIT_FAILURE);
  }

  // initialize the memory
  for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
    h_odata[i] = (unsigned char)(i & 0xff);
  }

  for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
    h_cacheClear1[i] = (unsigned char)(i & 0xff);
    h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
  }

  // allocate device memory
  unsigned char *d_idata;
  checkCudaErrors(cudaMalloc((void **)&d_idata, memSize));

  // warmup
  for (int r = 0; r < nWarmups; r++) {
    checkCudaErrors(
      cudaMemcpy(d_idata, h_odata, memSize, cudaMemcpyHostToDevice));
  }

  // copy host memory to device memory
  if (PINNED == memMode) {
    if (bDontUseGPUTiming) sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < nRepeats; i++) {
      checkCudaErrors(cudaMemcpyAsync(d_idata, h_odata, memSize,
                                      cudaMemcpyHostToDevice, 0));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
    if (bDontUseGPUTiming) {
      sdkStopTimer(&timer);
      elapsedTimeInMs = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
    }
  } else {
    elapsedTimeInMs = 0;
    for (unsigned int i = 0; i < nRepeats; i++) {
      sdkStartTimer(&timer);
      checkCudaErrors(
          cudaMemcpy(d_idata, h_odata, memSize, cudaMemcpyHostToDevice));
      sdkStopTimer(&timer);
      elapsedTimeInMs += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      memset(flush_buf, i, FLUSH_SIZE);
    }
  }

  // calculate bandwidth in GB/s
  double time_s = elapsedTimeInMs / 1e3;
  bandwidthInGBs = (memSize * (float)nRepeats) / (double)1e9;
  bandwidthInGBs = bandwidthInGBs / time_s;
  // clean up memory
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaEventDestroy(start));
  sdkDeleteTimer(&timer);

  if (PINNED == memMode) {
    checkCudaErrors(cudaFreeHost(h_odata));
  } else {
    free(h_odata);
  }

  free(h_cacheClear1);
  free(h_cacheClear2);
  checkCudaErrors(cudaFree(d_idata));

  return bandwidthInGBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a device to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
extern "C" float testDeviceToDeviceTransfer(unsigned int memSize, unsigned int nWarmups, 
                                            unsigned int nRepeats) {
  StopWatchInterface *timer = NULL;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInGBs = 0.0f;
  cudaEvent_t start, stop;

  sdkCreateTimer(&timer);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // allocate host memory
  unsigned char *h_idata = (unsigned char *)malloc(memSize);

  if (h_idata == 0) {
    fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
    exit(EXIT_FAILURE);
  }

  // initialize the host memory
  for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
    h_idata[i] = (unsigned char)(i & 0xff);
  }

  // allocate device memory
  unsigned char *d_idata;
  checkCudaErrors(cudaMalloc((void **)&d_idata, memSize));
  unsigned char *d_odata;
  checkCudaErrors(cudaMalloc((void **)&d_odata, memSize));

  // initialize memory
  checkCudaErrors(
      cudaMemcpy(d_idata, h_idata, memSize, cudaMemcpyHostToDevice));

  // warmup
  for (int r = 0; r < nWarmups; r++) {
    checkCudaErrors(
      cudaMemcpy(d_odata, d_idata, memSize, cudaMemcpyDeviceToDevice));
  }

  // run the memcopy
  sdkStartTimer(&timer);
  checkCudaErrors(cudaEventRecord(start, 0));

  for (unsigned int i = 0; i < nRepeats; i++) {
    checkCudaErrors(
        cudaMemcpy(d_odata, d_idata, memSize, cudaMemcpyDeviceToDevice));
  }

  checkCudaErrors(cudaEventRecord(stop, 0));

  // Since device to device memory copies are non-blocking,
  // cudaDeviceSynchronize() is required in order to get
  // proper timing.
  checkCudaErrors(cudaDeviceSynchronize());

  // get the total elapsed time in ms
  sdkStopTimer(&timer);
  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

  if (bDontUseGPUTiming) {
    elapsedTimeInMs = sdkGetTimerValue(&timer);
  }

  // calculate bandwidth in GB/s
  double time_s = elapsedTimeInMs / 1e3;
  bandwidthInGBs = (2.0f * memSize * (float)nRepeats) / (double)1e9;
  bandwidthInGBs = bandwidthInGBs / time_s;

  // clean up memory
  sdkDeleteTimer(&timer);
  free(h_idata);
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));

  return bandwidthInGBs;
}