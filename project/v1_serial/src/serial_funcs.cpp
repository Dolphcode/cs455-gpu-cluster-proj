#include <cmath>

#include "serial_funcs.h"

void conv_layer_serial(tensor3_t *in, tensor3_t *out, conv_t *kernel, int padding) {
	// Setup the output tensor
	out->w = 2 * padding + ((in->w - 2 * kernel->padding) / kernel->stride);
	out->h = 2 * padding + ((in->h - 2 * kernel->padding) / kernel->stride);
	out->c = kernel->filters;
	out->data = (float*)(out + 1);

	// Perform the convolution AND the sigmoid iteratively
	for (int filter = 0; filter < kernel->filters; filter++) {
		// Get the address of the current kernel we're using to compute
		float *curr_kernel = &kernel->kernel[
			filter * (kernel->dim * kernel->dim + 1)
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
								(y_pix - kernel_y_offset) * in->w +
								(x_pix - kernel_x_offset)
							];

							// Add to sum
							sum += in_pix * curr_kernel[
								z * kernel->dim * kernel->channels +
								y * kernel->dim +
								x
							];
						}
					}
				}

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
