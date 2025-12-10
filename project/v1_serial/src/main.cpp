#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "prepostproc.h"
#include "layer_structs.h"
#include "serial_funcs.h"

#define 	MAX_TENSOR_BLOCK	(1280 * 1280 * 32 + 8)
#define		PREALLOC_TENSORS	20

#define		MB_UNIT			(2 << 20)

#define		MAX_LAYERS		50

//#define		THRESHOLD		0.9
//#define		IOU_THRESHOLD	0.3

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
	// Read from argv
	float THRESHOLD = atof(argv[1]);
	float IOU_THRESHOLD = atof(argv[2]);
	char fpath_buf[512];
	strcpy(fpath_buf, argv[3]);

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
	tensor3_t* img_tensor = load_image(fpath_buf, 1, blocks[0]);
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
	test_tensor_printf((tensor3_t*)blocks[1], 1, 1, 0);
	test_tensor_printf((tensor3_t*)blocks[1], 2, 1, 0);
	test_tensor_printf((tensor3_t*)blocks[1], 1, 2, 0);
	test_tensor_printf((tensor3_t*)blocks[1], 1, 1, 1);
	test_tensor_printf((tensor3_t*)blocks[1], 2, 1, 1);
	test_tensor_printf((tensor3_t*)blocks[1], 1, 2, 1);

	// Test the second layer
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[1], (tensor3_t*)blocks[2], conv_kernels[1], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 0, 0, 0);
	test_tensor_printf((tensor3_t*)blocks[2], 1, 0, 0);

	// Test the first c2f
	printf("\nComputing a C2f\n");
	c2f_layer_serial(
			(tensor3_t**)(blocks + 2),
			&conv_kernels[2],
			1,
			1,
			1);
	printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 1, 1, 0);
	test_tensor_printf((tensor3_t*)blocks[2], 2, 1, 0);
	
	// Conv
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[2], (tensor3_t*)blocks[0], conv_kernels[6], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[0], 0, 0, 0);
	test_tensor_printf((tensor3_t*)blocks[0], 1, 0, 0);
	
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
	test_tensor_printf((tensor3_t*)blocks[0], 2, 1, 0);

	// Clean out
	memset((tensor3_t*)blocks[1], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	// Conv
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[0], (tensor3_t*)blocks[1], conv_kernels[13], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[1], 0, 0, 0);
	test_tensor_printf((tensor3_t*)blocks[1], 1, 0, 0);
	
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
	test_tensor_printf((tensor3_t*)blocks[1], 2, 1, 0);
	
	// Clean out
	memset((tensor3_t*)blocks[2], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	// Conv
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[1], (tensor3_t*)blocks[2], conv_kernels[20], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[2], 0, 0, 0);
	test_tensor_printf((tensor3_t*)blocks[2], 1, 0, 0);
	
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
	test_tensor_printf((tensor3_t*)blocks[2], 2, 1, 0);
	
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
	
	// Concat into C2f
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
	tensor3_t* medium_head_out = c2f_layer_serial(&blocks[5], &conv_kernels[36], 1, 1, 0);
	test_tensor_printf(concat_out, 1, 1, 0);
	
	// Clean out
	memset(blocks[6], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Conv layer
	printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[5], (tensor3_t*)blocks[6], conv_kernels[40], 0, 0);
	printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf((tensor3_t*)blocks[6], 0, 0, 0);

	// Concat into C2f
	concat_out = (tensor3_t*)blocks[6];
	base_channels = concat_out->c;
	concat_out->c += small_bb_out->c;
	
	for (int c = 0; c < small_bb_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * small_bb_out->w * small_bb_out->h +
			row * small_bb_out->w + // This one has padding specifically
			col; // This one has padding specifically
			
		concat_out->data[out_idx] = small_bb_out->data[in_idx];
	}

	printf("\nC2f for the small sized head output\n");
	tensor3_t* small_head_out = c2f_layer_serial(&blocks[6], &conv_kernels[41], 1, 1, 0);
	test_tensor_printf(concat_out, 1, 1, 0);

	// Clean out
	memset(blocks[7], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	
	printf("\nOccupied Addresses:\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n",
			large_bb_out, blocks[0],
			med_bb_out, blocks[1],
			small_bb_out, blocks[2],
			medium_h1_out, blocks[3],
			large_h1_out, blocks[4],
			medium_head_out, blocks[5],
			small_head_out, blocks[6]);

	// Detect 1
	printf("\nBBox Detect 64 Channel\n");
	printf("Input: %d %d %d\n", large_h1_out->w, large_h1_out->h, large_h1_out->c);
	detect_layer_serial(large_h1_out, &blocks[7], &conv_kernels[45]);
	memset(blocks[8], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(blocks[7], 0, 0, 0);

	// Detect 2
	printf("\nBBox Detect 128 Channel\n");
	printf("Input: %d %d %d\n", medium_head_out->w, medium_head_out->h, medium_head_out->c);
	detect_layer_serial(medium_head_out, &blocks[8], &conv_kernels[48]);
	memset(blocks[9], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(blocks[8], 0, 0, 0);

	// Detect 3
	printf("\nBBox Detect 256 Channel\n");
	printf("Input: %d %d %d\n", small_head_out->w, small_head_out->h, small_head_out->c);
	detect_layer_serial(small_head_out, &blocks[9], &conv_kernels[51]);
	memset(blocks[10], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(blocks[9], 0, 0, 0);

	// Detect 1
	printf("\nClass Detect 64 Channel\n");
	printf("Input: %d %d %d\n", large_h1_out->w, large_h1_out->h, large_h1_out->c);
	detect_layer_serial(large_h1_out, &blocks[10], &conv_kernels[54]);
	memset(blocks[11], 0, 4 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(blocks[10], 0, 0, 0);

	// Detect 2
	printf("\nClass Detect 128 Channel\n");
	printf("Input: %d %d %d\n", medium_head_out->w, medium_head_out->h, medium_head_out->c);
	detect_layer_serial(medium_head_out, &blocks[11], &conv_kernels[57]);
	memset(blocks[12], 0, 4 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(blocks[11], 0, 0, 0);
	test_tensor_printf(blocks[11], 1, 0, 0);
	test_tensor_printf(blocks[11], 2, 0, 0);

	// Detect 3
	printf("\nBBox Detect 256 Channel\n");
	printf("Input: %d %d %d\n", small_head_out->w, small_head_out->h, small_head_out->c);
	detect_layer_serial(small_head_out, &blocks[12], &conv_kernels[60]);
	memset(blocks[13], 0, 4 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(blocks[12], 0, 0, 0);

	// -------------------------------------
	// 	OUTPUT BBOXES
	// -------------------------------------
	
	// Allocate bounding box memory
	int total_boxes = blocks[7]->w * blocks[7]->h + 
		blocks[8]->w * blocks[8]->h +
		blocks[9]->w * blocks[9]->h;
	bbox_t *boxes = (bbox_t*)calloc(total_boxes, sizeof(bbox_t));

	float bins[16];
	float classes[80];
	float softmax_sum, expected;
	int box_center_x, box_center_y, stride;
	stride = PROCESS_DIM / blocks[7]->h / 2; // Just use h for now honestly, assume square so no matter
	
	// Iterate
	for (int y = 0; y < blocks[7]->h; y++)
	for (int x = 0; x < blocks[7]->w; x++) {

		softmax_sum = 0.0;
		expected = 0.0;

		box_center_x = x * stride * 2 + stride;
		box_center_y = y * stride * 2 + stride;

		// Start with the left pixel
		for (int c = 0; c < 16; ++c) {
			bins[c] = blocks[7]->data[
				c * blocks[7]->w * blocks[7]->h +
				y * blocks[7]->w +
				x
			];

			softmax_sum += exp(bins[c]);
		}
		for (int c = 0; c < 16; ++c) {
			bins[c] = exp(bins[c]) / softmax_sum;
			expected += bins[c] * c;
		}
		boxes[y * blocks[7]->w + x].x = box_center_x - int(round(expected * stride * 2));
				
		softmax_sum = 0.0;
		expected = 0.0;
		// Then the top pixel
		for (int c = 0; c < 16; ++c) {
			bins[c] = blocks[7]->data[
				(c + 16) * blocks[7]->w * blocks[7]->h +
				y * blocks[7]->w +
				x
			];

			softmax_sum += exp(bins[c]);
		}
		for (int c = 0; c < 16; ++c) {
			bins[c] = exp(bins[c]) / softmax_sum;
			expected += bins[c] * c;
		}
		boxes[y * blocks[7]->w + x].y = box_center_y - int(round(expected * stride * 2));
	
		softmax_sum = 0.0;
		expected = 0.0;
		// Then the right pixel
		for (int c = 0; c < 16; ++c) {
			bins[c] = blocks[7]->data[
				(c + 32) * blocks[7]->w * blocks[7]->h +
				y * blocks[7]->w +
				x
			];

			softmax_sum += exp(bins[c]);
		}
		for (int c = 0; c < 16; ++c) {
			bins[c] = exp(bins[c]) / softmax_sum;
			expected += bins[c] * c;
		}
		boxes[y * blocks[7]->w + x].w = box_center_x + int(round(expected * stride * 2)) - boxes[y * blocks[7]->w + x].x;
	
		softmax_sum = 0.0;
		expected = 0.0;
		// Then the top pixel
		for (int c = 0; c < 16; ++c) {
			bins[c] = blocks[7]->data[
				(c + 48) * blocks[7]->w * blocks[7]->h +
				y * blocks[7]->w +
				x
			];

			softmax_sum += exp(bins[c]);
		}
		for (int c = 0; c < 16; ++c) {
			bins[c] = exp(bins[c]) / softmax_sum;
			expected += bins[c] * c;
		}
		boxes[y * blocks[7]->w + x].h = box_center_y + int(round(expected * stride * 2)) - boxes[y * blocks[7]->w + x].y;
	
		/*
		printf("Box Detected:\n(%d, %d) %dx%d\n", 
				boxes[y * blocks[7]->w + x].x,
				boxes[y * blocks[7]->w + x].y,
				boxes[y * blocks[7]->w + x].w,
				boxes[y * blocks[7]->w + x].h
		      );*/

		// Class Selection
		softmax_sum = 0.0;
		expected = 0.0;
		for (int c = 0; c < 80; ++c) {
			classes[c] = blocks[10]->data[
				c * blocks[10]->w * blocks[10]->h +
				y * blocks[10]->w +
				x
			];

			softmax_sum += exp(classes[c]);
		}
		for (int c = 0; c < 80; ++c) {
			classes[c] = 1.0f / (1.0f + expf(-classes[c]));
			if (classes[c] > expected) {
				expected = classes[c];
				boxes[y * blocks[7]->w + x].cid = c;
				boxes[y * blocks[7]->w + x].class_conf = classes[c];
			}
		}


	}	

	int offset = blocks[7]->w * blocks[7]->h;
	stride = PROCESS_DIM / blocks[8]->h / 2; // Just use h for now honestly, assume square so no matter


	// Iterate
	for (int y = 0; y < blocks[8]->h; y++)
		for (int x = 0; x < blocks[8]->w; x++) {

			softmax_sum = 0.0;
			expected = 0.0;

			box_center_x = x * stride * 2 + stride;
			box_center_y = y * stride * 2 + stride;

			// Start with the left pixel
			for (int c = 0; c < 16; ++c) {
				bins[c] = blocks[8]->data[
					c * blocks[8]->w * blocks[8]->h +
					y * blocks[8]->w +
					x
				];

				softmax_sum += exp(bins[c]);
			}
			for (int c = 0; c < 16; ++c) {
				bins[c] = exp(bins[c]) / softmax_sum;
				expected += bins[c] * c;
			}
			boxes[offset + y * blocks[8]->w + x].x = box_center_x - int(round(expected * stride * 2));

			softmax_sum = 0.0;
			expected = 0.0;
			// Then the top pixel
			for (int c = 0; c < 16; ++c) {
				bins[c] = blocks[8]->data[
					(c + 16) * blocks[8]->w * blocks[8]->h +
					y * blocks[8]->w +
					x
				];

				softmax_sum += exp(bins[c]);
			}
			for (int c = 0; c < 16; ++c) {
				bins[c] = exp(bins[c]) / softmax_sum;
				expected += bins[c] * c;
			}
			boxes[offset + y * blocks[8]->w + x].y = box_center_y - int(round(expected * stride * 2));

			softmax_sum = 0.0;
			expected = 0.0;
			// Then the right pixel
			for (int c = 0; c < 16; ++c) {
				bins[c] = blocks[8]->data[
					(c + 32) * blocks[8]->w * blocks[8]->h +
					y * blocks[8]->w +
					x
				];

				softmax_sum += exp(bins[c]);
			}
			for (int c = 0; c < 16; ++c) {
				bins[c] = exp(bins[c]) / softmax_sum;
				expected += bins[c] * c;
			}
			boxes[offset + y * blocks[8]->w + x].w = box_center_x + int(round(expected * stride * 2)) - boxes[offset + y * blocks[8]->w + x].x;

			softmax_sum = 0.0;
			expected = 0.0;
			// Then the top pixel
			for (int c = 0; c < 16; ++c) {
				bins[c] = blocks[8]->data[
					(c + 48) * blocks[8]->w * blocks[8]->h +
					y * blocks[8]->w +
					x
				];

				softmax_sum += exp(bins[c]);
			}
			for (int c = 0; c < 16; ++c) {
				bins[c] = exp(bins[c]) / softmax_sum;
				expected += bins[c] * c;
			}
			boxes[offset + y * blocks[8]->w + x].h = box_center_y + int(round(expected * stride * 2)) - boxes[offset + y * blocks[8]->w + x].y;

			/*
			printf("Box Detected:\n(%d, %d) %dx%d\n",
				   boxes[y * blocks[8]->w + x].x,
		  boxes[y * blocks[8]->w + x].y,
		  boxes[y * blocks[8]->w + x].w,
		  boxes[y * blocks[8]->w + x].h
			);*/

			// Class Selection
			softmax_sum = 0.0;
			expected = 0.0;
			for (int c = 0; c < 80; ++c) {
				classes[c] = blocks[11]->data[
					c * blocks[11]->w * blocks[11]->h +
					y * blocks[11]->w +
					x
				];

				softmax_sum += exp(classes[c]);
			}
			for (int c = 0; c < 80; ++c) {
				classes[c] = 1.0f / (1.0f + expf(-classes[c]));
				if (classes[c] > expected) {
					expected = classes[c];
					boxes[offset + y * blocks[8]->w + x].cid = c;
					boxes[offset + y * blocks[8]->w + x].class_conf = classes[c];
				}
			}
		}

		offset += blocks[8]->w * blocks[8]->h;
		stride = PROCESS_DIM / blocks[9]->h / 2; // Just use h for now honestly, assume square so no matter


		// Iterate
		for (int y = 0; y < blocks[9]->h; y++)
			for (int x = 0; x < blocks[9]->w; x++) {

				softmax_sum = 0.0;
				expected = 0.0;

				box_center_x = x * stride * 2 + stride;
				box_center_y = y * stride * 2 + stride;

				// Start with the left pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[9]->data[
						c * blocks[9]->w * blocks[9]->h +
						y * blocks[9]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}
				boxes[offset + y * blocks[9]->w + x].x = box_center_x - int(round(expected * stride * 2));

				softmax_sum = 0.0;
				expected = 0.0;
				// Then the top pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[9]->data[
						(c + 16) * blocks[9]->w * blocks[9]->h +
						y * blocks[9]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}
				boxes[offset + y * blocks[9]->w + x].y = box_center_y - int(round(expected * stride * 2));

				softmax_sum = 0.0;
				expected = 0.0;
				// Then the right pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[9]->data[
						(c + 32) * blocks[9]->w * blocks[9]->h +
						y * blocks[9]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}
				boxes[offset + y * blocks[9]->w + x].w = box_center_x + int(round(expected * stride * 2)) - boxes[offset + y * blocks[9]->w + x].x;

				softmax_sum = 0.0;
				expected = 0.0;
				// Then the top pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[9]->data[
						(c + 48) * blocks[9]->w * blocks[9]->h +
						y * blocks[9]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}
				boxes[offset + y * blocks[9]->w + x].h = box_center_y + int(round(expected * stride * 2)) - boxes[offset + y * blocks[9]->w + x].y;

				// Class Selection
				softmax_sum = 0.0;
				expected = 0.0;
				for (int c = 0; c < 80; ++c) {
					classes[c] = blocks[12]->data[
						c * blocks[12]->w * blocks[12]->h +
						y * blocks[12]->w +
						x
					];

					softmax_sum += exp(classes[c]);
				}
				for (int c = 0; c < 80; ++c) {
					classes[c] = 1.0f / (1.0f + expf(-classes[c]));
					if (classes[c] > expected) {
						expected = classes[c];
						boxes[offset + y * blocks[9]->w + x].cid = c;
						boxes[offset + y * blocks[9]->w + x].class_conf = classes[c];
					}
				}
			}

	// Initialize inactive flags and build list of candidate indices
	std::vector<int> idxs;
	idxs.reserve(total_boxes);
	for (int i = 0; i < total_boxes; ++i) {
		if (boxes[i].class_conf >= THRESHOLD) {
			boxes[i].inactive = 0;
			idxs.push_back(i);
		} else {
			boxes[i].inactive = 1;
		}
	}

	// Sort by confidence high -> low
	std::sort(idxs.begin(), idxs.end(),
			  [&](int a, int b) {
				  return boxes[a].class_conf > boxes[b].class_conf;
			  });

	// Standard NMS: for each box, suppress later boxes with high IoU and same class
	for (size_t m = 0; m < idxs.size(); ++m) {
		int i = idxs[m];
		if (boxes[i].inactive) continue;

		for (size_t n = m + 1; n < idxs.size(); ++n) {
			int j = idxs[n];
			if (boxes[j].inactive) continue;

			// Optional: per-class NMS
			if (boxes[i].cid != boxes[j].cid) continue;

			float x_int_min = std::max(boxes[i].x, boxes[j].x);
			float y_int_min = std::max(boxes[i].y, boxes[j].y);
			float x_int_max = std::min(boxes[i].x + boxes[i].w, boxes[j].x + boxes[j].w);
			float y_int_max = std::min(boxes[i].y + boxes[i].h, boxes[j].y + boxes[j].h);
			float inter_area = std::max(0.0f, x_int_max - x_int_min) * std::max(0.0f, y_int_max - y_int_min);

			float area1 = boxes[i].w * boxes[i].h;
			float area2 = boxes[j].w * boxes[j].h;
			float union_area = area1 + area2 - inter_area;

			float iou = inter_area / union_area;
			if (iou > IOU_THRESHOLD) {
				boxes[j].inactive = 1;
			}
		}
	}

	cv::Mat image = cv::imread(fpath_buf, cv::IMREAD_COLOR);
	//cv::Size newsize(PROCESS_DIM, PROCESS_DIM);
	//cv::Mat resized;
	//cv::resize(image, resized, newsize);
	float r = std::min((float)PROCESS_DIM / (float)image.cols, (float)PROCESS_DIM / (float)image.rows);
	float pad_w = PROCESS_DIM - (r * image.cols);
	float pad_h = PROCESS_DIM - (r * image.rows);

	// Debug print: how many were suppressed
	int disabled = 0;
	for (int i = 0; i < total_boxes; ++i) {
		if (boxes[i].inactive) disabled++;

		if (!boxes[i].inactive) {
			int x1 = std::round((boxes[i].x - (pad_w / 2)) / r);
			int y1 = std::round((boxes[i].y - (pad_h / 2)) / r);
			int x2 = std::round((boxes[i].w / r) + x1);
			int y2 = std::round((boxes[i].h / r) + y1);

			cv::rectangle(
				image,
				 cv::Point(x1, y1- 20),
						  cv::Point(x1 + 100, y1),
						  cv::Scalar(255, 0, 0),
						  -1
			);

			cv::rectangle(
				image,
				 cv::Point(x1, y1),
						  cv::Point(x2, y2),
						  cv::Scalar(255, 0, 0),
						  2
			);

			char out_text[256];
			sprintf(out_text, "id%d, %.2f", boxes[i].cid, boxes[i].class_conf);

			cv::putText(
				image,
			   out_text,
			   cv::Point(x1, y1),
						cv::FONT_HERSHEY_SIMPLEX,
			   0.5,
			   cv::Scalar(255, 255, 255)
			);

			/*
			cv::rectangle(
				resized,
				cv::Point(boxes[i].x, boxes[i].y - 20),
				cv::Point(boxes[i].x + 100, boxes[i].y),
				cv::Scalar(255, 0, 0),
				-1
			);

			cv::rectangle(
				resized,
				cv::Point(boxes[i].x, boxes[i].y),
				cv::Point(boxes[i].x + boxes[i].w, boxes[i].y + boxes[i].h),
				cv::Scalar(255, 0, 0),
				2
			);

			char out_text[256];
			sprintf(out_text, "id%d, %.2f", boxes[i].cid, boxes[i].class_conf);

			cv::putText(
				resized,
				out_text,
				cv::Point(boxes[i].x, boxes[i].y),
				cv::FONT_HERSHEY_SIMPLEX,
				0.5,
				cv::Scalar(255, 255, 255)
			);
			*/
		}


	}
	cv::imwrite("output.jpg", image);
	printf("Disabled %d boxes\n", disabled);

	// Cleanup
	printf("Finishing work\n", infile);	
	free(boxes);
	fclose(infile);
	free(conv_buf);
	free(img_buf);
}
