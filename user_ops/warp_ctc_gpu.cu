#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "warp_ctc.h"

// Simple wrapper for warp-ctc on the gpu
void warp_ctc_gpu(const float* const inputs, float* grads, 
                  const int* const labels, 
                  const int* const label_lengths, 
                  const int* const input_lengths, 
                  int num_classes, int batch_size, 
                  float *losses) {

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
            "Error making cuda stream.");

    ctcOptions info;
    info.loc = CTC_GPU;
    info.stream = stream;
    info.blank_label = 0;

    size_t gpu_alloc_bytes;
    get_workspace_size(label_lengths, input_lengths,
                       num_classes, batch_size, info,
                       &gpu_alloc_bytes);

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "Error allocating memory");

    throw_on_error(compute_ctc_loss(inputs, grads, labels,
                                    label_lengths, input_lengths,
                                    num_classes, batch_size,
                                    losses, ctc_gpu_workspace, info),
                   "Error computing CTC loss.");

    cudaFree(ctc_gpu_workspace);
}
