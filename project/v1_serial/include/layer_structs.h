#ifndef __LAYER_STRUCTS__
#define __LAYER_STRUCTS__

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
	float 	*kernel;
}conv_t;

/**
 * A struct defining a generic 4D tensor
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
 * Initializes a convolutional layer at the pointer ptr
 */
void conv_layer(int dim, int channels, int stride, int padding, int filters, void *ptr);

#endif 
