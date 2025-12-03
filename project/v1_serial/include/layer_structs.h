#ifndef __LAYER_STRUCTS__
#define __LAYER_STRUCTS__

#include <cstdio>

/**
 * A struct defining metadata for a convolutional layer assuming a square kernel
 * The actual kernel data will be parsed under the assumption that kernel points
 * to a contiguous block of memory containing an alternating sequence of flattened
 * kernels and their corresponding biases:
 * 	filter0, bias0, filter1, bias1, ...
 */
typedef struct {
	int	dim;
	int	channels;
	int	stride;
	int	padding;
	int	filters;
	int	data_len;
	float*	kernel;
}conv_t;

/**
 * Computes the amount of space to allocate for a convolutional layer and returns 
 * the result.
 */
size_t conv_malloc_amt(
		int dim,
		int channels,
		int filters
);

/**
 * Reads a convolutional layer from a binary file to a buffer. Returns the address
 * after the convolution layer in memory.
 */
void* fread_conv(
		FILE *infile, // The binary file to read from
		void *buf // The buffer to write
);

/**
 * A 3D tensor type struct containing all the metadata to be able to parse the 
 * tensor
 */
typedef struct {
	int w;
	int h;
	int c;
	float *data;
}tensor3_t;

#endif
