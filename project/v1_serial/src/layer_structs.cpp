#include <cstdlib>

#include "layer_structs.cpp"


void conv_malloc_amt(int dim, int channels, int filters, int padding, int w, int h, int batch_size, size_t *conv_size, size_t *layer_in_size) {
	// Compute the convolution memory allocation amount
	if (conv_size) {
		*conv_size += sizeof(conv_t);

		// Each filter is dim x dim x channels
		// Each filter also has a bias term, hence + 1
		*conv_size += (dim * dim * channels + 1) * filters * sizeof(float);
	}

	// Computes the memory allocation amount for the input layer
	if (layer_in_size) {
		*layer_in_size += sizeof(tensor_t);

		// Each image is w x h x channels
		//  + 2 x padding on each side
		// Assume images are lined up vertically in a single minibatch
		// So the total height should be h x batch_size + (batch_size + 1) * padding
		*layer_in_size += (w + 2 * padding) * 
			(h * batch_size + (batch_size + 1) * padding) * 
			channels * 
			sizeof(float);
	}
}

void* conv_layer(int dim, int channels, int stride, int padding, int filters, void *ptr) {
	// Cast the address and ensure it is valid
	conv_t *conv_layer = (conv_t*)ptr;
	if (!conv_layer) 
		return NULL;

	// Set all the values
	conv_layer->dim = dim;
	conv_layer->channels = channels;
	conv_layer->stride = stride;
	conv_layer->filters = filters;

	// Get the data pointer
	float *data = (float*)(conv_layer + 1);
	conv_layer->kernel = data;

	// Now compute the pointer after the data
	void *next = (void*)(data + (dim * dim * channels + 1) * filters);
}
