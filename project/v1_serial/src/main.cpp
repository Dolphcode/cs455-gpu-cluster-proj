#include <iostream>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "prepostproc.h"
#include "layer_structs.h"
#include "serial_funcs.h"

#define 	MAX_TENSOR_BLOCK	(1280 * 1280 * 32 + 8)
#define		PREALLOC_TENSORS	20

#define		MB_UNIT			(2 << 20)

#define		MAX_LAYERS		50

void test_tensor_printf(tensor3_t* tensor, int x, int y, int channel) {
	printf("Tensor is %d x %d with %d channels\n",
			tensor->w,
			tensor->h,
			tensor->c);

	
	printf("\tValue at (%d, %d, %d) is %f\n",
			channel,
			x,
			y,
			tensor->data[
				channel * tensor->w * tensor->h + 
				y * tensor->w + 
				x
			]
		);

}

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
	
	
	printf("Allocating %d MiB of memory for conv layers\n\n", conv_block_size / MB_UNIT);
	void *conv_buf = calloc(conv_block_size / sizeof(float), sizeof(float));

	// Load the image
	tensor3_t* img_tensor = load_image("./deer.jpg", 1, blocks[0]);
	printf("Image successfully loaded!\n");

	// Load the convolution layers
	FILE *infile = fopen("./filters.bin", "rb");
	void *curr = conv_buf;
	int counter = 0;

	conv_t *conv_kernels[64];

	int curr_kernel = 0;	
	while (1) {
		// Exit out if we hit the EOF
		if (curr == NULL) break;

		// Save the kernel pointer
		conv_kernels[curr_kernel] = (conv_t*)curr;
		curr = fread_conv(infile, curr);
		
		/* DEBUG Print conv kernels as we read them	
		printf("dim=%d, stride=%d, pad=%d, filters=%d\n\n", conv_kernels[curr_kernel]->dim, 
				conv_kernels[curr_kernel]->stride, 
				conv_kernels[curr_kernel]->padding, 
				conv_kernels[curr_kernel]->filters);
		*/

		curr_kernel++;
		counter++;
	}
	printf("Read %d convolutional layers from the binary file\n", counter);
	
	// Test the first layer
	conv_t *conv_lay = (conv_t*)conv_buf;
	printf("\nComputing a convolution\n");
	conv_layer_serial(img_tensor, (tensor3_t*)blocks[1], conv_lay, 1, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[1], 1, 1, 1);

	// Test the second layer
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[1], (tensor3_t*)blocks[2], conv_kernels[1], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 1, 1, 1);

	// Test the first c2f
	printf("\nComputing a C2f\n");
	c2f_layer_serial(
			(tensor3_t**)(blocks + 2),
			&conv_kernels[2],
			1,
			1,
			1);
	printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 0, 0, 0);
	
	// Conv
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[2], (tensor3_t*)blocks[0], conv_kernels[6], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[0], 0, 0, 0);
	
	// Clean out
	memset((tensor3_t*)blocks[1], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Test the first c2f
	printf("\nComputing a C2f\n");
	tensor3_t* large_bb_out = c2f_layer_serial(
			(tensor3_t**)(blocks),
			&conv_kernels[7],
			2,
			1,
			1);
	printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf((tensor3_t*)blocks[0], 1, 1, 0);

	// Clean out
	memset((tensor3_t*)blocks[1], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	// Conv
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[0], (tensor3_t*)blocks[1], conv_kernels[13], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[1], 0, 0, 0);
	
	// Test the first c2f
	printf("\nComputing a C2f\n");
	tensor3_t* med_bb_out = c2f_layer_serial(
			(tensor3_t**)(blocks + 1),
			&conv_kernels[14],
			2,
			1,
			1);
	printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf((tensor3_t*)blocks[1], 1, 1, 0);
	
	// Clean out
	memset((tensor3_t*)blocks[2], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	// Conv
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[1], (tensor3_t*)blocks[2], conv_kernels[20], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 0, 0, 0);
	
	// Test the first c2f
	printf("\nComputing a C2f\n");
	c2f_layer_serial(
			(tensor3_t**)(blocks + 2),
			&conv_kernels[21],
			1,
			0,
			1);
	printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 0, 0, 0);
	
	// Clean out
	memset((tensor3_t*)blocks[3], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	// Test the SPPF
	printf("\nPerforming SPPF\n");
	tensor3_t* small_bb_out = sppf_layer_serial(
			(tensor3_t**)(blocks + 2),
			&conv_kernels[25],
			0
	);
	printf("Completed! Here's some data about the output of sppf:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 0, 0, 0);
	
	// Clean out
	memset((tensor3_t*)blocks[3], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	// Upsample + Concat
	int base_channels;
	tensor3_t* concat_out;

	printf("\nUpsampling and Concatenating\n");
	concat_out = (tensor3_t*)blocks[3];
	upsample_layer_serial(small_bb_out, concat_out);
	base_channels = concat_out->c;
	concat_out->c += med_bb_out->c;

	for (int c = 0; c < med_bb_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * med_bb_out->w * med_bb_out->h +
			(row + 1) * med_bb_out->w + // This one has padding specifically
			(col + 1); // This one has padding specifically
			
		concat_out->data[out_idx] = med_bb_out->data[in_idx];
	}
	test_tensor_printf(concat_out, 0, 0, 256);

	printf("\nC2f for the first medium sized head output\n");
	tensor3_t* medium_h1_out = c2f_layer_serial(&blocks[3], &conv_kernels[27], 1, 0, 0);
	test_tensor_printf(concat_out, 1, 1, 0);

	// Clean out
	memset(blocks[4], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
		
	// Upsample + Concat
	printf("\nUpsampling and Concatenating\n");
	concat_out = (tensor3_t*)blocks[4];
	upsample_layer_serial(medium_h1_out, concat_out);
	base_channels = concat_out->c;
	concat_out->c += large_bb_out->c;

	for (int c = 0; c < large_bb_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * large_bb_out->w * large_bb_out->h +
			(row + 1) * large_bb_out->w + // This one has padding specifically
			(col + 1); // This one has padding specifically
		
		concat_out->data[out_idx] = large_bb_out->data[in_idx];
	}
	test_tensor_printf(concat_out, 0, 0, 0);

	printf("\nC2f for the first large sized head output\n");
	tensor3_t* large_h1_out = c2f_layer_serial(&blocks[4], &conv_kernels[31], 1, 1, 0);
	test_tensor_printf(large_h1_out, 1, 1, 0);

	// Conv layer
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[4], (tensor3_t*)blocks[5], conv_kernels[35], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[5], 0, 0, 0);

	concat_out = (tensor3_t*)blocks[5];
	base_channels = concat_out->c;
	concat_out->c += medium_h1_out->c;
	
	for (int c = 0; c < medium_h1_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * medium_h1_out->w * medium_h1_out->h +
			row * medium_h1_out->w + // This one has padding specifically
			col; // This one has padding specifically
			
		concat_out->data[out_idx] = medium_h1_out->data[in_idx];
	}

	printf("\nC2f for the medium sized head output\n");
	conv_kernels[36]->stride = 2; // TEMP FIX
	tensor3_t* medium_head_out = c2f_layer_serial(&blocks[5], &conv_kernels[36], 1, 1, 0);
	test_tensor_printf(concat_out, 1, 1, 0);
	
	

	// Clean out
	memset(blocks[6], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	printf("\nOccupied Addresses:\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n",
			large_bb_out, blocks[0],
			med_bb_out, blocks[1],
			small_bb_out, blocks[2],
			medium_h1_out, blocks[3],
			large_h1_out, blocks[4]);
	
	// Cleanup
	printf("Finishing work\n", infile);	
	fclose(infile);
	free(conv_buf);
	free(img_buf);
}
