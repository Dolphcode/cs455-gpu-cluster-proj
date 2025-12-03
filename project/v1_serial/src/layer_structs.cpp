#include "layer_structs.h"

size_t conv_malloc_amt(int dim, int channels, int filters) {
	// Initialize to 0
	size_t alloc_amt = 0;
	
	// Space for the convolution metadata
	alloc_amt += sizeof(conv_t);

	// Space for the filters themselves
	alloc_amt += (dim * dim * channels * filters) * sizeof(float);

	//
	alloc_amt += filters;

	return alloc_amt;
}

void* fread_conv(FILE *infile, void *buf) {
	// Validate the file and the buffer
	if (!infile || !buf) return NULL;

	// Read from the file into buffer
	fread(buf, sizeof(int), 5, infile);

	// Convert the buf pointer to a convolution pointer
	conv_t *metadata = (conv_t*)buf;

	// Compute the data length
	metadata->data_len = (metadata->dim * metadata->dim * metadata->channels + 1) * metadata->filters;
	
	// Set the data pointer
	metadata->kernel = (float*)(metadata + 1);

	// Read into the buffer
	fread(metadata->kernel, sizeof(float), metadata->data_len, infile);

	// Return the pointer after
	return (void*)(metadata->kernel + metadata->data_len);	
}
