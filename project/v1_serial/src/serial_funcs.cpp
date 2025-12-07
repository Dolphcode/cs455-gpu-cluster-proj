#include <cmath>

#include "serial_funcs.h"

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
/*
// Assume concat with no padding because concat always comes before c2f
void concat_serial(tensor3_t **tensor_buf, tensor3_t *out, int count) {
	// Find total # of channels
	int total_channels = 0;
	for (int i = 0; i < count; ++i) total_channels += tensor_buf[i]->c;
	
	// Setup output
	out->w = tensor_buf[0]->w - (2 * tensor_buf[0]->padding)
	out->h = tensor_buf[0]->h - (2 * tensor_buf[0]->padding)
	out->c = total_channels;
	out->data = (float*)(out + 1);
	
	int cur_tensor = 0;
	for (int c = 0; c < total_channels; ++c) {
		

	       	for (int row = 0; row < out->h; ++row) {
			float* in_base_addr = 
	       	}
	}	       
}

void c2f_layer_serial(tensor3_t **tensor_buf, conv_t **conv_buf, int n, int out_padding) {
	// Track the running total of convolutional layers parsed
	int convs_parsed = 0;
	int tensors_parsed = 0;
	
	// First convolutional layer
	conv_layer_serial(tensor_buf[tensors_parsed], tensor_buf[tensors_parsed + 1], conv_buf[convs_parsed++], 1, 0);
	tensors_parsed++;

	// Split is implicit. Share memory in the in buffer, extract the latter half
	tensor_buf[tensors_parsed]->c = tensor_buf[tensors_parsed]->c / 2;
	tensor_buf[tensors_parsed]->data = (float*)(tensor_buf[tensors_parsed] + 1) + 
		(tensor_buf[tensors_parsed]->w * tensor_buf[tensors_parsed]->h * tensor_buf[tensors_parsed]->c);

	int bottleneck_start = tensors_parsed;
	for (int i = 0; i < n; i++) {
		conv_layer_serial(tensor_buf[tensors_parsed], tensor_buf[tensors_parsed + 1], conv_buf[convs_parsed++], 1, 0);
		tensors_parsed++;
		conv_layer_serial(tensor_buf[tensors_parsed], tensor_buf[tensors_parsed + 1], conv_buf[convs_parsed++], 1, 0);
		tensors_parsed++;
	}

	tensor3_t **buf = (tensor3_t**)malloc(sizeof(tensor3_t*) * (1 + n)); // (The first two are in one tensor so I'll just concatenate as if that was just the one)
	buf[0] = tensor_buf[bottleneck_start];
	for (int i = 0; i < n; i++) {
		buf[i + 1] = tensor_buf[bottleneck_start + 2 * (i + 1)];
	}
	concat_serial(buf, tensor_buf[tensors_parsed + 1], 1 + n); 
	tensors_parsed++;
	
	// The final convolution
	conv_layer_serial(tensor_buf[tensors_parsed], tensor_buf[tensors_parsed + 1], conv_buf[convs_parsed++], out_padding, 0);	
}
*/
