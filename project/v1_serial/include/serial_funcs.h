#ifndef __SERIAL_FUNCS__
#define __SERIAL_FUNCS__

#include "layer_structs.h"

/**
 * One convolution layer computed serially. The output buffer will be modified 
 * to produce the correct output in the correct format.
 */
void conv_layer_serial(
		tensor3_t *in, 	// The input tensor 
		tensor3_t *out, // The output tensor
		conv_t *kernel, // The kernel to be used
		int padding, 	// Padding in the output tensor if needed
		int keep_tensor
);


void c2f_layer_serial(
		tensor3_t *in,
		tensor3_t *temp,
		tensor3_t *out,
 		conv_t **conv_buf,
		int n,
		int out_padding
);

void concat_serial(
		tensor3_t **tensor_buf,
		tensor3_t *out,
		int count
);

void set_padding_serial(
		tensor3_t *in, 
		tensor3_t *out, 
		int from, 
		int to
);

#endif
