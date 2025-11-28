#ifndef __LAYER_STRUCTS__
#define __LAYER_STRUCTS__

#define 	MAX_KERNEL_SIZE 	7
#define		MAX_CHANNELS		512

/**
 * A struct defining metadata for a convolutional layer assuming a square kernel
 */
typedef struct {
	int 	dim; // The W x H dimension of a single filter
	int 	channels; // The number of channels per filter
	int 	stride; // The stride for this convolutional layer
	int 	padding; // The padding in the input tensor for this layer
	int	filters; // The number of filters in this layer
	float 	*kernel;
}conv_t;

/**
 * A struct defining a generic 3D tensor
 */
typedef struct {
	int depth;
	int w;
	int h;
	float *data;
}tensor_t;

#endif 
