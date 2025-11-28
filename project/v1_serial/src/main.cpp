#include <iostream>

#include "serial_funcs.h"

int main(int argc, char *argv[]) {
	// Compute the kernel block size
	// for testing let's just use the first two conv layers of YOLOv8
	size_t block_size = sizeof(conv_t);
	block_size += sizeof(float) * 3 * 3 * 3 * 16; // The filters themselves
	block_size += sizeof(float) * 16; // The bias terms
	
	block_size += sizeof(conv_t);
	block_size += sizeof(float) * 16 * 3 * 3 * 32;
	block_size += sizeof(float) * 32;


	void *kernel_block = calloc(block_size / sizeof(float), sizeof(float)); // This should be divisible by 4

	free(kernel_block);	
}
