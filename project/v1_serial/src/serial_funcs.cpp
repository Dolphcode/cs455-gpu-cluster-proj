#include <iostream>

#include <cmath>

#include "serial_funcs.h"


void conv_2d_serial_forward(tensor_t *in, tensor_t *out, conv_t *kernel) {

	// STEP ONE: Convolution
	// Separate kernel into filters
	for (int filter = 0; filter < kernel->filters; filter++) {
		float *curr_kernel = &kernel->kernel[filter * (kernel->dim * kernel->dim + 1)];
		
		// TODO: Detect if there's padding in the output image or not based on
		// 	 its dimension
		//
		// TODO: Handle batches in the output image if padding is needed

		printf("Working on filter %d\n", filter);
		for (int i = 1; i < out->h - 1; i++) {
			for (int j = 1; j < out->w - 1; j++) {
				float *out_pix = &out->data[i * out->w * out->d + j * out->d + filter];
				float sum = 0.0;

				for (int z = 0; z < kernel->channels; z++) {
					for (int y = 0; y < kernel->dim; y++) {
						for (int x = 0; x < kernel->dim; x++) {
							int x_pix = ((j - kernel->padding) * kernel->stride) + kernel->padding;
							int y_pix = ((i - kernel->padding) * kernel->stride) + kernel->padding;
							int kernel_x_offset = x - (kernel->dim / 2);
							int kernel_y_offset = y - (kernel->dim / 2);
							float in_pix = in->data[(y_pix - kernel_y_offset) * in->d * in->w + (x_pix - kernel_x_offset) * in->d + z];
							sum += in_pix * curr_kernel[z * kernel->dim * kernel->channels + y * kernel->dim + x];
						}
					}
				}

				// Add the bias term
				sum += curr_kernel[kernel->dim * kernel->dim * kernel->channels];
				*out_pix = sum;
			}
		}
	}

	// STEP TWO: Compute mean and stdev per output channel for Batch Norm
	// TODO: I think this works with padding assuming padding is zeros
	float *means = (float*)calloc(kernel->filters, sizeof(float)), *stdevs = (float*)calloc(kernel->filters, sizeof(float));
	int tot_elems = out->w * out->h * out->d;
	int elems_per_channel = out->w * out->h;
	// Means
	for (int elem = 0; elem < tot_elems; elem++) {
		means[elem % kernel->filters] += out->data[elem] / elems_per_channel;
	}
	// Standard Deviations
	for (int elem = 0; elem < tot_elems; elem++) {
		stdevs[elem % kernel->filters] += (out->data[elem] - means[elem % kernel->filters]) * (out->data[elem] - means[elem % kernel->filters]);
	}
	for (int i = 0; i < kernel->filters; i++) {
		stdevs[i] /= elems_per_channel - 1;
		stdevs[i] = sqrt(stdevs[i]);
	}

	// Apply batch norm and SiLU
	for (int elem = 0; elem < tot_elems; elem++) {
		if (out->data[elem] == 0.0) continue; // Ignore padding
		out->data[elem] = (out->data[elem] - means[elem % kernel->filters]) / stdevs[elem % kernel->filters];
		float sigmoid = 1.0 / (1.0 + exp(-out->data[elem]));
		out->data[elem] *= sigmoid;
	}

}
