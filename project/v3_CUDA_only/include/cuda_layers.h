#ifndef CUDA_LAYERS_H
#define CUDA_LAYERS_H

#include "layer_structs.h"

void conv_layer_cuda(
		tensor3_t *in,
		tensor3_t *out,
		conv_t *kernel,
		int padding,
		int keep_tensor
		);

#endif
