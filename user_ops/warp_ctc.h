#pragma once

#include <ctc.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>


void warp_ctc_gpu(const float* const inputs, float* grads, 
                  const int* const labels, 
                  const int* const label_lengths, 
                  const int* const input_lengths, 
                  int num_classes, int batch_size, 
                  float *losses);

inline 
void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
        throw std::runtime_error(message + 
                (", stat = " + std::string(ctcGetStatusString(status))));
    }
}

inline 
void throw_on_error(cudaError_t error, const char* message) {
    if (error) {
        throw thrust::system_error(error, 
                thrust::cuda_category(), message);
    }
}



