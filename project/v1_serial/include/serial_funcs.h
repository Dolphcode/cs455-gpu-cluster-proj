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
		int padding 	// Padding in the output tensor if needed
);


#endif
