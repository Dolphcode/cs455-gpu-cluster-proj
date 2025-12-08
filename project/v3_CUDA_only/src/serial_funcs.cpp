#include <cmath>
#include <cstring>

#include "serial_funcs.h"
#include "cuda_layers.h"

void conv_layer_serial(tensor3_t *in, tensor3_t *out, conv_t *kernel, int padding, int keep_tensor) {
	// Checks
	printf("Currently working on kernel with %d %d %d %d %d\n",
			kernel->dim,
			kernel->channels,
			kernel->filters,
			kernel->stride,
			kernel->padding);
	
	// Setup the output tensor
	if (!keep_tensor) {
		out->w = 2 * padding + ((in->w - 2 * kernel->padding) / kernel->stride);
		out->h = 2 * padding + ((in->h - 2 * kernel->padding) / kernel->stride);
		out->c = kernel->filters;
		out->data = (float*)(out + 1);
	}

	// Perform the convolution AND the sigmoid iteratively
	for (int filter = 0; filter < kernel->filters; filter++) {
		// Get the address of the current kernel we're using to compute
		float *curr_kernel = &kernel->kernel[
			filter * (kernel->dim * kernel->dim * kernel->channels + 1)
		];
		
		// Work from the output
		for (int out_row = padding; out_row < out->h - padding; out_row++) {
			for (int out_col = padding; out_col < out->w - padding; out_col++) {
				// Zero initialize the sum
				float sum = 0.0;

				// Compute the pixels in the input we're pulling from
				int x_pix = ((out_col - padding) * kernel->stride) + kernel->padding;
				int y_pix = ((out_row - padding) * kernel->stride) + kernel->padding;
				
				// Perform the kernel convolution
				for (int z = 0; z < kernel->channels; z++) {
					for (int y = 0; y < kernel->dim; y++) {
						for (int x = 0; x < kernel->dim; x++) {
							// Compute the offset from the current input pixel we're centering the convolution around
							int kernel_x_offset = x - (kernel->dim / 2);
							int kernel_y_offset = y - (kernel->dim / 2);

							// Grab the input pixel
							float in_pix = in->data[
								z * in->w * in->h +
								(y_pix + kernel_y_offset) * in->w +
								(x_pix + kernel_x_offset)
							];
							if (in_pix > 100) printf("Conv IN VAL ERR %f %d %d\n", in_pix, in->w, in->h);

							// Add to sum
							sum += in_pix * curr_kernel[
								z * kernel->dim * kernel->dim +
								y * kernel->dim +
								x
							];
								
						}
					}
				}
					
				if (sum > 100) printf("Conv ERROR: %f\n", sum);
				// Add the bias
				sum += kernel->kernel[(kernel->dim * kernel->dim * kernel->channels + 1) * filter 
							+ (kernel->dim * kernel->dim * kernel->channels)];

				// Compute the sigmoid
				float sigmoid = 1.0 / (1.0 + exp(-sum));

				// Set the output pixel to the sum
				out->data[
					filter * out->w * out->h +
					out_row * out->w +
					out_col
				] = sum * sigmoid;
			}
		}
	}
}

tensor3_t* c2f_layer_serial(tensor3_t **tensor_buf, conv_t **conv_buf, int n, int out_padding, int shortcut) {
	// Find the total # of channels

	// Compute the first layer with padding because the bottlenecks require padding	
	conv_layer_cuda(tensor_buf[0], tensor_buf[1], conv_buf[0], 1, 0);

	// Split is implicit
	
	tensor_buf[1]->c /= 2;
	tensor_buf[1]->data = (float*)(tensor_buf[1] + 1) + 
		(tensor_buf[1]->w * tensor_buf[1]->h * tensor_buf[1]->c);

	// Bottleneck
	int i;
	for (i = 0; i < n; i++) {
		int a = 1 + i * 2;
		int b = a + 1, c = a + 2;
		conv_layer_cuda(tensor_buf[a], tensor_buf[b], conv_buf[1 + i * 2], 1, 0);
		conv_layer_cuda(tensor_buf[b], tensor_buf[c], conv_buf[2 + i * 2], 1, 0);
		if (shortcut) {
			for (int chan = 0; chan < tensor_buf[c]->c; chan++) {
				for (int row = 1; row < tensor_buf[a]->h - 1; row++) {
					for (int col = 1; col < tensor_buf[a]->w - 1; col++ ) {
						tensor_buf[c]->data[
							chan * tensor_buf[c]->w * tensor_buf[c]->h +
							(row) * tensor_buf[c]->w +
							(col)
						] 
						 +=
						tensor_buf[a]->data[
							chan * tensor_buf[a]->w * tensor_buf[a]->h +
							row * tensor_buf[a]->w +
							col
						];
					}
				}
			}	
		}
	}

	// Prepare the concatenation output tensor
	int out_offset = 3 + (n - 1) * 2 + 1;
	tensor3_t *concat_out = tensor_buf[out_offset];
	concat_out->w = tensor_buf[out_offset - 1]->w - 2;
	concat_out->h = tensor_buf[out_offset - 1]->h - 2;
	concat_out->c = tensor_buf[out_offset - 1]->c * (n + 2);
	concat_out->data = (float*)(concat_out + 1);

	// Reset the input tensor to the bottleneck
	tensor_buf[1]->c *= 2;
	tensor_buf[1]->data = (float*)(tensor_buf[1] + 1);

	int in_c = 0;
	int curr_tensor_idx = 1;
	tensor3_t* curr_tensor = tensor_buf[curr_tensor_idx];
	for (int out_c = 0; out_c < tensor_buf[out_offset - 1]->c * (n + 2); out_c++) {
		// Update the current tensor
		if (in_c >= curr_tensor->c) {
			in_c = 0;
			curr_tensor_idx += 2;
			curr_tensor = tensor_buf[curr_tensor_idx];
		}

		for (int out_row = 0; out_row < tensor_buf[out_offset - 1]->h; out_row++) {
			for (int out_col = 0; out_col < tensor_buf[out_offset - 1]->w; out_col++) {
				concat_out->data[
					out_c * concat_out->w * concat_out->h +
					out_row * concat_out->w +
					out_col
				]
				 =
				curr_tensor->data[
					in_c * curr_tensor->w * curr_tensor->h +
					(out_row + 1) * curr_tensor->w +
					(out_col + 1)
				];
			}
		}

		// Update in_c
		in_c++;
	}

	// Final convolution but also zero out padding first
	memset(tensor_buf[0]->data, 0, sizeof(float) * (out_padding * 2 + concat_out->w) * (out_padding * 2 + concat_out->h) * conv_buf[2 * n + 1]->filters); 
							
	conv_layer_cuda(concat_out, tensor_buf[0], conv_buf[2 * n + 1], out_padding, 0);
	
	return tensor_buf[0];
}

tensor3_t *sppf_layer_serial(tensor3_t **tensor_buf, conv_t **conv_buf, int out_padding) {
	// First convolution layer
	conv_layer_cuda(tensor_buf[0], tensor_buf[1], conv_buf[0], 2, 0);

	// Max Pool Series!!!
	for (int i = 0; i < 3; i++) {
		// Prepare the tensors
		tensor3_t *in_t = tensor_buf[1 + i];
		tensor3_t *out_t = tensor_buf[2 + i];
		out_t->c = in_t->c;
		out_t->w = in_t->w;
		out_t->h = in_t->h;
		out_t->data = (float*)(out_t + 1);

		// Perform the max pool operation
		for (int c = 0; c < in_t->c; c++)
		for (int in_row = 2; in_row < in_t->h - 2; in_row++)
		for (int in_col = 2; in_col < in_t->w - 2; in_col++) {

			// Initialize the running max
			float running_max = in_t->data[
				c * in_t->w * in_t->h +
				(in_row - 2) * in_t->w +
				(in_col - 2)
			];

			// Find the max
			for (int y_offset = -2; y_offset < 3; y_offset++) 
			for (int x_offset = -2; x_offset < 3; x_offset++) {
				float test_val = in_t->data[
					c * in_t->w * in_t->h +
					(in_row + y_offset) * in_t->w +
					(in_col + x_offset)
				];
				
				if (test_val > running_max) running_max = test_val;
			}

			// Set the max in the output
			out_t->data[
				c * out_t->w * out_t->h +
				in_row * out_t->w +
				in_col
			] = running_max;
		}	
	}

	// Setup the concatenation output
	tensor3_t* concat_out = tensor_buf[5];
	concat_out->w = tensor_buf[1]->w - 4;
	concat_out->h = tensor_buf[1]->h - 4;
	concat_out->c = tensor_buf[1]->c * 4;
	concat_out->data = (float*)(concat_out + 1);
	
	// Concat the results of the max pools with no padding
	for (int i = 0; i < 4; i++) {
		// Select the current in to concat
		tensor3_t* curr_in = tensor_buf[1 + i];
		
		// Concatenate
		for (int c = 0; c < curr_in->c; c++)
		for (int row = 0; row < concat_out->h; row++)
		for (int col = 0; col < concat_out->w; col++) {
			int out_index = (i * curr_in->c + c) * concat_out->w * concat_out->h +
				row * concat_out->w +
				col;

			int in_index = c * curr_in->w * curr_in->h +
				(row + 2) * curr_in->w +
				(col + 2);
			if (concat_out->data[out_index] > 100.0)
				printf("FLAGGED %f!\n", concat_out->data[out_index]);
			concat_out->data[out_index] = curr_in->data[in_index];
		}
	}

	// Final convolution
	conv_layer_cuda(concat_out, tensor_buf[0], conv_buf[1], out_padding, 0);
	return tensor_buf[0];
}

void upsample_layer_serial(tensor3_t *in, tensor3_t *out) {
	// Determine output size (Assuming no padding)
	out->w = in->w * 2;
	out->h = in->h * 2;
	out->c = in->c;
	out->data = (float*)(out + 1);

	// Upsampling
	for (int c = 0; c < in->c; c++)
	for (int row = 0; row < in->h; row++)
	for (int col = 0; col < in->w; col++) {
		// Pull the value
		float val = in->data[
			c * in->w * in->h +
			row * in->w +
			col
		];

		// Upsample
		int out_base = c * out->w * out->h +
			(2 * row) * out->w +
			(2 * col);
		if (val > 100) printf("%f %p %p %d %d %d\n", val, in->data, in->data + c * in->w * in->h + row * in->w + col, row, col, c);
		// Replicate into 4 to double the image size
		out->data[out_base] = val;
		out->data[out_base + 1] = val;
		out->data[out_base + out->w] = val;
		out->data[out_base + out->w + 1] = val;
	}
}

tensor3_t* detect_layer_serial(tensor3_t *in, tensor3_t **tensor_buf, conv_t **conv_buf) {
	// First two convolutional layers
	conv_layer_cuda(in, tensor_buf[0], conv_buf[0], 1, 0);
	conv_layer_cuda(tensor_buf[0], tensor_buf[1], conv_buf[1], 0, 0);
	
	// Set
	conv_t *kernel = conv_buf[2];
	tensor3_t *out = tensor_buf[0];
	in = tensor_buf[1];
	int padding = 0;

	// Checks
	printf("Currently working on kernel with %d %d %d %d %d\n",
			kernel->dim,
			kernel->channels,
			kernel->filters,
			kernel->stride,
			kernel->padding);
	
	// Setup the output tensor
	out->w = 2 * padding + ((in->w - 2 * kernel->padding) / kernel->stride);
	out->h = 2 * padding + ((in->h - 2 * kernel->padding) / kernel->stride);
	out->c = kernel->filters;
	out->data = (float*)(out + 1);

	// Perform the convolution AND the sigmoid iteratively
	for (int filter = 0; filter < kernel->filters; filter++) {
		// Get the address of the current kernel we're using to compute
		float *curr_kernel = &kernel->kernel[
			filter * (kernel->dim * kernel->dim * kernel->channels + 1)
		];
		
		// Work from the output
		for (int out_row = padding; out_row < out->h - padding; out_row++) {
			for (int out_col = padding; out_col < out->w - padding; out_col++) {
				// Zero initialize the sum
				float sum = 0.0;

				// Compute the pixels in the input we're pulling from
				int x_pix = ((out_col - padding) * kernel->stride) + kernel->padding;
				int y_pix = ((out_row - padding) * kernel->stride) + kernel->padding;
				
				// Perform the kernel convolution
				for (int z = 0; z < kernel->channels; z++) {
					for (int y = 0; y < kernel->dim; y++) {
						for (int x = 0; x < kernel->dim; x++) {
							// Compute the offset from the current input pixel we're centering the convolution around
							int kernel_x_offset = x - (kernel->dim / 2);
							int kernel_y_offset = y - (kernel->dim / 2);

							// Grab the input pixel
							float in_pix = in->data[
								z * in->w * in->h +
								(y_pix + kernel_y_offset) * in->w +
								(x_pix + kernel_x_offset)
							];
							if (in_pix > 100) printf("Conv IN VAL ERR %f %d %d\n", in_pix, in->w, in->h);

							// Add to sum
							sum += in_pix * curr_kernel[
								z * kernel->dim * kernel->dim +
								y * kernel->dim +
								x
							];
								
						}
					}
				}
					
				// Add the bias
				sum += kernel->kernel[(kernel->dim * kernel->dim * kernel->channels + 1) * filter 
							+ (kernel->dim * kernel->dim * kernel->channels)];

				// Set the output pixel to the sum
				out->data[
					filter * out->w * out->h +
					out_row * out->w +
					out_col
				] = sum;
			}
		}
	}

	return tensor_buf[0];
}
