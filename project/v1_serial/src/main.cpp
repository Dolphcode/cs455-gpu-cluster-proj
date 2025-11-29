#include <iostream>

#include "layer_structs.h"

int main(int argc, char *argv[]) {
	// Compute the kernel block size
	// for testing let's just use the first two conv layers of YOLOv8
	size_t block_size = 0, data_size = 0;
	conv_malloc_amt(3, 3, 16, 1, 640, 640, 1, &block_size, &data_size); // Conv 1
	conv_malloc_amt(3, 16, 32, 1, 320, 320, 1, &block_size, &data_size); // Conv 2

	printf("%dB, %dB to allocate\n", block_size, data_size);

	void *conv_ptr = calloc(block_size / sizeof(float), sizeof(float));
	void *data_ptr = calloc(data_size / sizeof(float), sizeof(float));
	void *curr = conv_ptr;

	printf("%x\n", curr);
	curr = conv_layer(3, 3, 2, 1, 16, curr);
	printf("%x\n", curr);
	curr = conv_layer(3, 16, 2, 1, 32, curr);

	free(conv_ptr);
	free(data_ptr);
	return 0;
}
