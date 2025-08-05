/*-----------------------------------------------------------------------*/
/* Program: p2pBandwidthTest                                             */
/* Original code: https://github.com/NVIDIA/cuda-samples                 */
/*                                                                       */
/* This program measures GPU memory transfer rates in GB/s for simple.   */
/*-----------------------------------------------------------------------*/

#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include <helper_timer.h>

using namespace std;

typedef enum {
  P2P_WRITE = 0,
  P2P_READ = 1,
} P2PDataTransfer;

typedef enum {
  CE = 0,
  SM = 1,
} P2PEngine;

P2PEngine p2p_mechanism = CE;  // By default use Copy Engine

extern "C" double* testBandwidthMatrix(int numElems, int numGPUs, bool p2p, 
    P2PDataTransfer p2p_method, unsigned int nWarmups, unsigned int nRepeats);
extern "C" double* testBidirectionalBandwidthMatrix(int numElems, int numGPUs, bool p2p,
    unsigned int nWarmups, unsigned int nRepeats);
extern "C" double* testLatencyMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method,
    unsigned int nWarmups, unsigned int nRepeats);

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

__global__ void delay(volatile int *flag,
                      unsigned long long timeout_clocks = 10000000) {
  // Wait until the application notifies us that it has completed queuing up the
  // experiment, or timeout and exit, allowing the application to make progress
  long long int start_clock, sample_clock;
  start_clock = clock64();

  while (!*flag) {
    sample_clock = clock64();

    if (sample_clock - start_clock > timeout_clocks) {
      break;
    }
  }
}

// This kernel is for demonstration purposes only, not a performant kernel for
// p2p transfers.
__global__ void copyp2p(int4 *__restrict__ dest, int4 const *__restrict__ src,
                        size_t num_elems) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;

#pragma unroll(5)
  for (size_t i = globalId; i < num_elems; i += gridSize) {
    dest[i] = src[i];
  }
}

void performP2PCopy(int *dest, int destDevice, int *src, int srcDevice,
                    int num_elems, int repeat, bool p2paccess,
                    cudaStream_t streamToRun) {
  int blockSize = 0;
  int numBlocks = 0;

  cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p);
  cudaCheckError();

  if (p2p_mechanism == SM && p2paccess) {
    for (int r = 0; r < repeat; r++) {
      copyp2p<<<numBlocks, blockSize, 0, streamToRun>>>(
          (int4 *)dest, (int4 *)src, num_elems / 4);
    }
  } else {
    for (int r = 0; r < repeat; r++) {
      cudaMemcpyPeerAsync(dest, destDevice, src, srcDevice,
                          sizeof(int) * num_elems, streamToRun);
    }
  }
}

extern "C" double* testBandwidthMatrix(int numElems, int numGPUs, bool p2p, P2PDataTransfer p2p_method,
                                       unsigned int nWarmups, unsigned int nRepeats) {
  volatile int *flag = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);
  vector<cudaStream_t> stream(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
    cudaMalloc(&buffers[d], numElems * sizeof(int));
    cudaCheckError();
    cudaMemset(buffers[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaMalloc(&buffersD2D[d], numElems * sizeof(int));
    cudaCheckError();
    cudaMemset(buffersD2D[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
  }

  vector<double> bandwidthMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaCheckError();
          cudaDeviceEnablePeerAccess(i, 0);
          cudaCheckError();
          cudaSetDevice(i);
          cudaCheckError();
        }
      }

      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      cudaCheckError();
      cudaEventRecord(start[i], stream[i]);
      cudaCheckError();

      if (i == j) {
        // Perform intra-GPU, D2D copies
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, nRepeats,
                       access, stream[i]);
      } else {
        if (p2p_method == P2P_WRITE) {
          performP2PCopy(buffers[j], j, buffers[i], i, numElems, nRepeats, access,
                         stream[i]);
        } else {
          performP2PCopy(buffers[i], i, buffers[j], j, numElems, nRepeats, access,
                         stream[i]);
        }
      }

      cudaEventRecord(stop[i], stream[i]);
      cudaCheckError();

      // Release the queued events
      *flag = 1;
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      float time_ms;
      cudaEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = numElems * sizeof(int) * nRepeats / (double)1e9;
      if (i == j) {
        gb *= 2;  // must count both the read and the write here
      }
      bandwidthMatrix[i * numGPUs + j] = gb / time_s;
      if (p2p && access) {
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
        cudaSetDevice(i);
        cudaCheckError();
      }
    }
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream[d]);
    cudaCheckError();
  }

  cudaFreeHost((void *)flag);
  cudaCheckError();

  double* result = new double[bandwidthMatrix.size()];
  copy(bandwidthMatrix.begin(), bandwidthMatrix.end(), result);
  return result;
}

extern "C" double* testBidirectionalBandwidthMatrix(int numElems, int numGPUs, bool p2p,
                                                    unsigned int nWarmups, unsigned int nRepeats) {
  volatile int *flag = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);
  vector<cudaStream_t> stream0(numGPUs);
  vector<cudaStream_t> stream1(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaMalloc(&buffers[d], numElems * sizeof(int));
    cudaMemset(buffers[d], 0, numElems * sizeof(int));
    cudaMalloc(&buffersD2D[d], numElems * sizeof(int));
    cudaMemset(buffersD2D[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
    cudaStreamCreateWithFlags(&stream0[d], cudaStreamNonBlocking);
    cudaCheckError();
    cudaStreamCreateWithFlags(&stream1[d], cudaStreamNonBlocking);
    cudaCheckError();
  }

  vector<double> bandwidthMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaSetDevice(i);
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaDeviceEnablePeerAccess(i, 0);
          cudaCheckError();
        }
      }

      cudaSetDevice(i);
      cudaStreamSynchronize(stream0[i]);
      cudaStreamSynchronize(stream1[j]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      cudaSetDevice(i);
      // No need to block stream1 since it'll be blocked on stream0's event
      delay<<<1, 1, 0, stream0[i]>>>(flag);
      cudaCheckError();

      // Force stream1 not to start until stream0 does, in order to ensure
      // the events on stream0 fully encompass the time needed for all
      // operations
      cudaEventRecord(start[i], stream0[i]);
      cudaStreamWaitEvent(stream1[j], start[i], 0);

      if (i == j) {
        // For intra-GPU perform 2 memcopies buffersD2D <-> buffers
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, nRepeats,
                       access, stream0[i]);
        performP2PCopy(buffersD2D[i], i, buffers[i], i, numElems, nRepeats,
                       access, stream1[i]);
      } else {
        if (access && p2p_mechanism == SM) {
          cudaSetDevice(j);
        }
        performP2PCopy(buffers[i], i, buffers[j], j, numElems, nRepeats, access,
                       stream1[j]);
        if (access && p2p_mechanism == SM) {
          cudaSetDevice(i);
        }
        performP2PCopy(buffers[j], j, buffers[i], i, numElems, nRepeats, access,
                       stream0[i]);
      }

      // Notify stream0 that stream1 is complete and record the time of
      // the total transaction
      cudaEventRecord(stop[j], stream1[j]);
      cudaStreamWaitEvent(stream0[i], stop[j], 0);
      cudaEventRecord(stop[i], stream0[i]);

      // Release the queued operations
      *flag = 1;
      cudaStreamSynchronize(stream0[i]);
      cudaStreamSynchronize(stream1[j]);
      cudaCheckError();

      float time_ms;
      cudaEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = 2.0 * numElems * sizeof(int) * nRepeats / (double)1e9;
      if (i == j) {
        gb *= 2;  // must count both the read and the write here
      }
      bandwidthMatrix[i * numGPUs + j] = gb / time_s;
      if (p2p && access) {
        cudaSetDevice(i);
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
      }
    }
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream0[d]);
    cudaCheckError();
    cudaStreamDestroy(stream1[d]);
    cudaCheckError();
  }

  cudaFreeHost((void *)flag);
  cudaCheckError();

  double* result = new double[bandwidthMatrix.size()];
  copy(bandwidthMatrix.begin(), bandwidthMatrix.end(), result);
  return result;
}

extern "C" double* testLatencyMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method,
                                     unsigned int nWarmups, unsigned int nRepeats) {
  int numElems = 4;  // perform 1-int4 transfer.
  volatile int *flag = NULL;
  StopWatchInterface *stopWatch = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
  vector<cudaStream_t> stream(numGPUs);
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  if (!sdkCreateTimer(&stopWatch)) {
    printf("Failed to create stop watch\n");
    exit(EXIT_FAILURE);
  }
  sdkStartTimer(&stopWatch);

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
    cudaMalloc(&buffers[d], sizeof(int) * numElems);
    cudaMemset(buffers[d], 0, sizeof(int) * numElems);
    cudaMalloc(&buffersD2D[d], sizeof(int) * numElems);
    cudaMemset(buffersD2D[d], 0, sizeof(int) * numElems);
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
  }

  vector<double> gpuLatencyMatrix(numGPUs * numGPUs);
  vector<double> cpuLatencyMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaDeviceEnablePeerAccess(i, 0);
          cudaSetDevice(i);
          cudaCheckError();
        }
      }
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      cudaCheckError();
      cudaEventRecord(start[i], stream[i]);

      sdkResetTimer(&stopWatch);
      if (i == j) {
        // Perform intra-GPU, D2D copies
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, nRepeats,
                       access, stream[i]);
      } else {
        if (p2p_method == P2P_WRITE) {
          performP2PCopy(buffers[j], j, buffers[i], i, numElems, nRepeats, access,
                         stream[i]);
        } else {
          performP2PCopy(buffers[i], i, buffers[j], j, numElems, nRepeats, access,
                         stream[i]);
        }
      }
      float cpu_time_ms = sdkGetTimerValue(&stopWatch);

      cudaEventRecord(stop[i], stream[i]);
      // Now that the work has been queued up, release the stream
      *flag = 1;
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      float gpu_time_ms;
      cudaEventElapsedTime(&gpu_time_ms, start[i], stop[i]);

      gpuLatencyMatrix[i * numGPUs + j] = gpu_time_ms * 1e3 / nRepeats;
      cpuLatencyMatrix[i * numGPUs + j] = cpu_time_ms * 1e3 / nRepeats;
      if (p2p && access) {
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
        cudaSetDevice(i);
        cudaCheckError();
      }
    }
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream[d]);
    cudaCheckError();
  }

  sdkDeleteTimer(&stopWatch);

  cudaFreeHost((void *)flag);
  cudaCheckError();

  double* result = new double[gpuLatencyMatrix.size()];
  copy(gpuLatencyMatrix.begin(), gpuLatencyMatrix.end(), result);
  return result;
}
