#ifndef __LAYER_STRUCTS__
#define __LAYER_STRUCTS__

#include <cstdio>

#define 	MAX_KERNEL_SIZE 	7
#define		MAX_CHANNELS		512

/**
 * A struct defining metadata for a convolutional layer assuming a square kernel
 */
typedef struct {
	int 	dim; 		// The W x H dimension of a single filter
	int 	channels; 	// The number of channels per filter
	int 	stride; 	// The stride for this convolutional layer
	int 	padding; 	// The padding in the input tensor for this layer
	int	filters; 	// The number of filters in this layer
	int	data_len;	// Easily accessible precomputed data length
	float 	*kernel;
}conv_t;

/**
 * A struct defining a MOSTLY generic 4D tensor
 * 	n is used for filter dimension for convolutional kernel weight gradients
 * 	n is used for batch size otherwise
 */
typedef struct {
	int n; 		// Specifically used for the weight gradient, the # of filters
	int d; 		// The depth or number of channels
	int h; 		// The width of the tensor
	int w; 		// The height of the tensor
	float *data; 	// Pointer to the data of the tensor
}tensor_t;

/**
 * Computes the amount of space to allocate for a conv2d layer and the input tensor and adds to
 * the ptr values sent in
 */
void conv_malloc_amt(int dim,
			int channels,
			int filters,
			int padding,
			int w,
			int h,
			int batch_size,
			size_t* conv_size,
			size_t* layer_in_size);

/**
 * Initializes a convolutional layer at the pointer ptr. Returns a pointer to the address
 * after the conv layer's kernel data in memory
 */
void* conv_layer(int dim, int channels, int stride, int padding, int filters, void *ptr);

/**
 * Writes a conv layer
 */
void write_conv(FILE *outfile, conv_t *layer);

/**
 * Reades conv layer data and metadata
 */
void read_conv(FILE *infile, conv_t *layer);


void* create_tensor(int w, int h, int d, int n, void* buf);

#endif 
