#include <cstio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <math.h>
#include "cuda_layers.h"
#include "layer_structs.h"

__global__ void conv2d_sigmoid_kernel(
			const 
		)

__global__ void conv_dev_cuda(tensor3_t* in, tensor3_t* out, conv_t* kernel, int padding, int keep_tensor){
	if (!keep_tensor) {
                out->w = 2 * padding + ((in->w - 2 * kernel->padding) / kernel->stride);
                out->h = 2 * padding + ((in->h - 2 * kernel->padding) / kernel->stride);
                out->c = kernel->filters;
                out->data = (float*)(out + 1);
        }








}	
