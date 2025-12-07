#include <iostream>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "prepostproc.h"
#include "layer_structs.h"
#include "serial_funcs.h"

#define 	MAX_TENSOR_BLOCK	(320 * 320 * 16 + 8)
#define		PREALLOC_TENSORS	10

#define		MB_UNIT			(2 << 20)

#define		MAX_LAYERS		50

int main(int argc, char *argv[]) {
	// Allocate a big buffer
	void *img_buf = calloc(MAX_TENSOR_BLOCK * PREALLOC_TENSORS, sizeof(float));
	
	// Partition the big buffer into tensor blocks we can reuse over and over
	tensor3_t *blocks[PREALLOC_TENSORS];
	for (int i = 0; i < PREALLOC_TENSORS; i++)
		blocks[i] = (tensor3_t*)((float*)img_buf + (MAX_TENSOR_BLOCK * i));
	printf("Allocated %d max size tensors, worth %d MiB\n",
			PREALLOC_TENSORS,
			MAX_TENSOR_BLOCK * sizeof(float) * PREALLOC_TENSORS / MB_UNIT);
		
	// Allocate memory for all the convolution layers
	size_t conv_block_size = 0;
	conv_block_size += conv_malloc_amt(3, 3, 16); 		// Conv2D 0
	conv_block_size += conv_malloc_amt(3, 16, 32);		// Conv2D 1
	conv_block_size += c2f_malloc_amt(32, 32, 1);		// C2f 2
	conv_block_size += conv_malloc_amt(3, 32, 64);		// Conv2D 3	
	conv_block_size += c2f_malloc_amt(64, 64, 2);		// C2f 4
	conv_block_size += conv_malloc_amt(3, 64, 128);		// Conv2D 5
	conv_block_size += c2f_malloc_amt(128, 128, 2);		// C2f 6
	conv_block_size += conv_malloc_amt(3, 128, 256);	// Conv2D 7
	conv_block_size += c2f_malloc_amt(256, 256, 1);		// C2f 8
	conv_block_size += sppf_malloc_amt(256);		// SPPF 9
	conv_block_size += c2f_malloc_amt(384, 128, 1);		// C2f 12
	conv_block_size += c2f_malloc_amt(192, 64, 1);		// C2f 15
	conv_block_size += conv_malloc_amt(3, 64, 64);		// Conv 16
	conv_block_size += c2f_malloc_amt(192, 128, 1);		// C2f 18
	conv_block_size += conv_malloc_amt(3, 128, 128);	// Conv 19
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21

	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	
	
	printf("Allocating %d MiB of memory for conv layers\n", conv_block_size / MB_UNIT);
	void *conv_buf = calloc(conv_block_size / sizeof(float), sizeof(float));

	// Load the image
	tensor3_t* img_tensor = load_image("./deer.jpg", 1, blocks[0]);
	printf("Image successfully loaded!\n");

	// Load the convolution layers
	FILE *infile = fopen("./filters.bin", "rb");
	void *curr = conv_buf;
	int counter = 0;

	conv_t *track = (conv_t*)conv_buf;	
	while ((curr = fread_conv(infile, curr)) != NULL) {
		printf("Read convolution %d\n", counter);
		conv_t *curr_conv_buf = (conv_t*)curr;
		printf("%d B consumed already!\n", (char*)curr - (char*)conv_buf);
		printf("dim=%d, stride=%d, pad=%d, filters=%d\n\n", track->dim, 
				track->stride, 
				track->padding, 
				track->filters);

		counter++;
		track = curr_conv_buf;
	}
	
	fread_conv(infile, curr);
	printf("Read %d convolutional layers from the binary file\n", counter);
	
	// Test the first layer
	conv_t *conv_lay = (conv_t*)conv_buf;
	printf("\nComputing a convolution\n");
	conv_layer_serial(img_tensor, (tensor3_t*)blocks[1], conv_lay, 1, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	printf("\tTensor is %d x %d with %d channels\n", blocks[1]->w, blocks[1]->h, blocks[1]->c);
	printf("\tValue at (1, 1, 0) is %f\n", blocks[1]->data[1 * blocks[1]->w * blocks[1]->h + 1 * blocks[1]->w + 2]);

	// Cleanup	
	printf("Finishing work\n", infile);	
	fclose(infile);
	free(conv_buf);
	free(img_buf);
}
