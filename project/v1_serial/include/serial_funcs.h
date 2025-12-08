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


tensor3_t* c2f_layer_serial(
		tensor3_t **tensor_buf,	// A contiguous array of tensors to work with
 		conv_t **conv_buf,	// A contiguous array of convolution layers to work with
		int n,			// The number of bottleneck layers
		int out_padding,	// The amount of output padding
		int shortcut		// Whether this layer should activate the shortcut or not
);

tensor3_t* sppf_layer_serial(
		tensor3_t **tensor_buf,
		conv_t **conv_buf,
		int out_padding
);

void upsample_layer_serial(
		tensor3_t *in,
		tensor3_t *out
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
