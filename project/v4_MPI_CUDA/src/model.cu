#include "model.h"

#include "preprocess.h"

// Globally accessible device memory pointers
void *dev_kernel_buf; // Raw buffer for alloc and dealloc
void *dev_im2col_buf; // Raw buffer for alloc and dealloc
void *dev_blocks_buf; // Raw buffer for alloc and dealloc
conv_t *dev_kernels[NUM_LAYERS];
//tensor3_t *dev_im2col_block;
tensor3_t *dev_blocks[PREALLOC_TENSORS];
conv_t *h_kernels[NUM_LAYERS]; // We need access to the host kernel metadata for im2cole computation

// Global variables
int thds_per_blk;
int shared_mem;
float iou_thresh = DEFAULT_IOU_THRESH;
float conf_thresh = DEFAULT_CONF_THRESH;

__global__ void upsample(const tensor3_t *__restrict__ in, tensor3_t *__restrict__ out) {
	// Data Pointers
	const float *__restrict__ in_data = (float*)(in + 1);
	float *__restrict__ out_data = (float*)(out + 1);

	// Compute my coordinates
	const int channel = blockIdx.z;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Cache input values since we'll transfer them over to out later but also do the checks to see if this thread should even run
	// Really a bound guard
	const int in_w = in->w;
	const int in_h = in->h;
	const int in_c = in->c;
	const int out_w = in_w * 2;
	const int out_h = in_h * 2;
	if (x >= out_w || y >= out_h || channel >= in_c) return;

	// Set output tensor data
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && channel == 0) {
		out->w = out_w;
		out->h = out_h;
		out->c = in_c;
	}

	// Compute indexes
	int in_x = x / 2;
	int in_y = y / 2;
	int in_index = channel * in_w * in_h + in_y * in_w + in_x;
	int out_index = channel * out_w * out_h + y * out_w + x;

	// Transfer
	out_data[out_index] = in_data[in_index];
}

__global__ void maxpool2d_k5(const tensor3_t *__restrict__ in, tensor3_t *out) {
	// Data Pointers
	const float *__restrict__ in_data = (float*)(in + 1);
	float *__restrict__ out_data = (float*)(out + 1);

	// Compute my coordinates
	const int channel = blockIdx.z;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Cache input values since we'll transfer them over to out later
	const int in_w = in->w;
	const int in_h = in->h;
	const int in_c = in->c;
	if (x >= in_w || y >= in_h || channel >= in_c) return;

	// Set output tensor
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && channel == 0) {
		out->w = in_w;
		out->h = in_h;
		out->c = in_c;
	}

	// Compute the index of the output pixel we are assigning to the tensor
	const int out_index = channel * in_w * in_h +
		y * in_w +
		x;

	// Find the max from the 5 x 5 area
	float max = -FLT_MAX;
	for (int k_check_y = y - 2; k_check_y <= y + 2; k_check_y++)
	for (int k_check_x = x - 2; k_check_x <= x + 2; k_check_x++) {
		// Bound Check
		if (k_check_x < 0 || k_check_x >= in_w || k_check_y < 0 || k_check_y >= in_h) continue;

		// Compute index
		int check_index = channel * in_w * in_h +
			k_check_y * in_w +
			k_check_x;

		// Retrieve and check
		float check_val = in_data[check_index];
		if (check_val > max) max = check_val;
	}

	// Set the max
	out_data[out_index] = max;
}

__global__ void sum(const float *__restrict__ in, float *__restrict__ out, int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
		out[index] += in[index];
}

__global__ void conv_noswish(const tensor3_t *__restrict__ in, tensor3_t *__restrict__ out, const conv_t *__restrict__ kernel) {
	// Data Pointers
	const float *__restrict__ kernel_data = (float*)(kernel + 1);
	const float *__restrict__ in_data = (float*)(in + 1);
	float *__restrict__ out_data = (float*)(out + 1);

	// Store kernel and input tensor metadata in shared memory
	__shared__ tensor3_t in_meta;
	__shared__ conv_t kernel_meta;

	// Load to shared memory
	__shared__ float block_filter[KERNEL_MAX_FLOATS];
	int filter = blockIdx.z;
	int tid = blockDim.x * threadIdx.y + threadIdx.x;

	// Cache input data and kernels
	int out_w = in->w / kernel->stride;
	int out_h = in->h / kernel->stride;

	int floats_per_filter = kernel->dim * kernel->dim * kernel->channels + 1;
	int nThreads = blockDim.x * blockDim.y;
	for (int idx = tid; idx < floats_per_filter; idx += nThreads) {
		block_filter[idx] = kernel_data[idx + floats_per_filter * filter];
	}

	if (tid == 0 && filter == 0) { // Only a few threads in the first filter will set up the output tensor
		// This thread will set up the out tensor
		out->w = out_w;
		out->h = out_h;
		out->c = kernel->filters;
	}

	if (tid == 1) { // Also cache input tensor metadata and kernel metadata
		in_meta.w = in->w;
		in_meta.h = in->h;
		in_meta.c = in->c;

		kernel_meta.dim = kernel->dim;
		kernel_meta.stride = kernel->stride;
		kernel_meta.channels = kernel->channels;
		kernel_meta.filters = kernel->filters;
		kernel_meta.padding = kernel->padding;
	}

	__syncthreads();

	// Compute the index of the output pixel we are assigning to the tensor
	int out_index = blockIdx.z * out_w * out_h +
	(blockIdx.y * blockDim.y + threadIdx.y) * out_w +
	(blockIdx.x * blockDim.x + threadIdx.x);

	// Compute the center of the pixel we are responsible for
	int in_offset_y = (blockIdx.y * blockDim.y + threadIdx.y) * kernel_meta.stride;
	int in_offset_x = (blockIdx.x * blockDim.x + threadIdx.x) * kernel_meta.stride;

	// Store the sum
	float sum = 0.0f;
	for (int in_c = 0; in_c < kernel_meta.channels; in_c++) {
		// Iterate over the kernel
		for (int k_y = 0; k_y < kernel_meta.dim; k_y++)
			for (int k_x = 0; k_x < kernel_meta.dim; k_x++) {
				int kern_index = in_c * kernel_meta.dim * kernel_meta.dim + k_y * kernel_meta.dim + k_x;

				int k_offset_y = k_y - (kernel_meta.dim / 2);
				int k_offset_x = k_x - (kernel_meta.dim / 2);

				int in_y = (in_offset_y + k_offset_y);
				int in_x = (in_offset_x + k_offset_x);

				if (in_y < 0 || in_y >= in_meta.h || in_x < 0 || in_x >= in_meta.w) {
					continue;
				} else {
					int in_index = (in_meta.w * in_meta.h * in_c) + (in_y * in_meta.w) + in_x;
					sum += in_data[in_index] * block_filter[kern_index];
				}
			}
	}

	// Add the bias term for this filter
	sum += block_filter[kernel_meta.channels * kernel_meta.dim * kernel_meta.dim];

	// Set the output
	out_data[out_index] = sum;
}

__global__ void conv(const tensor3_t *__restrict__ in, tensor3_t *__restrict__ out, const conv_t *__restrict__ kernel) {
	const float *__restrict__ kernel_data = (float*)(kernel + 1);
	const float *__restrict__ in_data     = (float*)(in + 1);
	float       *__restrict__ out_data    = (float*)(out + 1);

	// Cache metadata into registers
	const int in_w   = in->w;
	const int in_h   = in->h;
	const int in_c   = in->c;

	const int k_dim      = kernel->dim;
	const int k_stride   = kernel->stride;
	const int k_channels = kernel->channels;
	const int k_filters  = kernel->filters;
	const int k_pad      = kernel->padding;  // currently unused in your math, but kept here

	// Compute output size same as your original
	const int out_w = in_w / k_stride;
	const int out_h = in_h / k_stride;

	// One thread per output element
	const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int out_c = blockIdx.z;

	// Bounds check
	if (out_x >= out_w || out_y >= out_h || out_c >= k_filters)
		return;

	// First thread writes metadata
	if (out_x == 0 && out_y == 0 && out_c == 0) {
		out->w = out_w;
		out->h = out_h;
		out->c = k_filters;
	}

	// Where this output sits in the input plane
	const int in_offset_y = out_y * k_stride;
	const int in_offset_x = out_x * k_stride;

	// Per-filter layout: [weights (k_channels * k_dim * k_dim)] [bias]
	const int floats_per_filter = k_dim * k_dim * k_channels + 1;
	const int filter_base       = floats_per_filter * out_c;

	float sum = 0.0f;

	for (int c = 0; c < k_channels; ++c) {
		for (int ky = 0; ky < k_dim; ++ky) {
			for (int kx = 0; kx < k_dim; ++kx) {
				int k_offset_y = ky - (k_dim / 2);
				int k_offset_x = kx - (k_dim / 2);

				int in_y = in_offset_y + k_offset_y;
				int in_x = in_offset_x + k_offset_x;

				if (in_x < 0 || in_x >= in_w || in_y < 0 || in_y >= in_h)
					continue;

				int in_index =
				c * (in_w * in_h) +
				in_y * in_w +
				in_x;

				int w_index =
				filter_base +
				c * (k_dim * k_dim) +
				ky * k_dim +
				kx;

				float w = kernel_data[w_index];
				float v = in_data[in_index];
				sum += v * w;
			}
		}
	}

	// Bias
	float bias = kernel_data[filter_base + k_dim * k_dim * k_channels];
	sum += bias;

	// Swish
	float sigmoid = 1.0f / (1.0f + expf(-sum));
	float out_val = sum * sigmoid;

	int out_index =
	out_c * (out_w * out_h) +
	out_y * out_w +
	out_x;

	out_data[out_index] = out_val;
}

void c2f(tensor3_t **tensors, conv_t** kernels, short n, short shortcut, dim3 grid_dim, dim3 block_dim, int rank) {
    // In case of errors
	cudaError_t err;

	// Compute the first convolution and extract the output tensor
	cudaMemset(tensors[1], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(tensors[0], tensors[1], kernels[0]);
	tensor3_t c2f_out;
	if ((err = cudaMemcpy(&c2f_out, tensors[1], sizeof(tensor3_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
		printf("Process %d: Failed to grab output tensor in C2f layer\n\t%s", rank, cudaGetErrorString(err));

	// Split but save the output size
	int out_channels = c2f_out.c;
	c2f_out.c = c2f_out.c / 2;
	if ((err = cudaMemcpy(tensors[1], &c2f_out, sizeof(tensor3_t), cudaMemcpyHostToDevice)) != cudaSuccess)
		printf("Process %d: Failed to reshape original tensor in C2f layer\n\t%s", rank, cudaGetErrorString(err));

	if ((err = cudaMemcpy(tensors[2], &c2f_out, sizeof(tensor3_t), cudaMemcpyHostToDevice)) != cudaSuccess) // Copy metadata to split tensor
		printf("Process %d: Failed to copy original tensor metadata to second split tensor in C2f layer\n\t%s", rank, cudaGetErrorString(err));

	int split_block_size = c2f_out.c * c2f_out.w * c2f_out.h; // Copy data into new tensor
	if ((err = cudaMemcpy((float*)(tensors[2] + 1), (float*)(tensors[1] + 1) + split_block_size, sizeof(float) * split_block_size,
			cudaMemcpyDeviceToDevice)) != cudaSuccess)
		printf("Process %d: Failed to copy split tensor data\n\t%s", rank, cudaGetErrorString(err));

	// Bottleneck
	grid_dim.z = c2f_out.c;
	for (int i = 0; i < n; i++) {
		cudaMemset(tensors[4 + i], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
		conv<<<grid_dim, block_dim>>>(tensors[2 + i], tensors[4 + i], kernels[1 + i * 2]);
		cudaMemset(tensors[3 + i], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
		conv<<<grid_dim, block_dim>>>(tensors[4 + i], tensors[3 + i], kernels[2 + i * 2]);

		if (shortcut) {
			int thread_count = c2f_out.c * c2f_out.w * c2f_out.h;
			sum<<<thread_count / 256, 256>>>((float*)(tensors[2 + i] + 1), (float*)(tensors[3 + i] + 1), thread_count);
		}
	}

	// Concat
	c2f_out.c *= (n + 2);
	if ((err = cudaMemcpy(tensors[3 + n], &c2f_out, sizeof(tensor3_t), cudaMemcpyHostToDevice)) != cudaSuccess)
		printf("Process %d: Failed to setup concat tensor\n\t%s", rank, cudaGetErrorString(err));

	for (int j = 0; j < n + 2; j++) {
		// Compute the output block offset
		int concat_offset = j * split_block_size;
		float* out_block = (float*)(tensors[3 + n] + 1) + concat_offset;

		// Copy each block to the concat tensor
		if ((err = cudaMemcpy(out_block, (float*)(tensors[1 + j] + 1), sizeof(float) * split_block_size, cudaMemcpyDeviceToDevice)) != cudaSuccess)
			printf("Process %d: Failed copy block %d to concat tensor\n\t%s", rank, j, cudaGetErrorString(err));
	}

	// Final conv -> Output into tensors 0
	grid_dim.z = out_channels;
	c2f_out.c = out_channels;
	if ((err = cudaMemcpy(tensors[0], &c2f_out, sizeof(tensor3_t), cudaMemcpyHostToDevice)) != cudaSuccess)
		printf("Process %d: Failed to copy correct tensor metadata\n\t%s", rank, cudaGetErrorString(err));
	cudaMemset(tensors[0], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(tensors[3 + n], tensors[0], kernels[1 + 2 * n]);
}

void sppf(tensor3_t **tensors, conv_t**kernels, conv_t **h_kernels, dim3 grid_dim, dim3 block_dim, int rank) {
	// In case of errors
	cudaError_t err;

	// Compute the first convolution
	grid_dim.z = h_kernels[0]->filters;
	cudaMemset(tensors[1], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(tensors[0], tensors[1], kernels[0]);

	// Cache the tensor data
	tensor3_t tensor_temp;
	cudaMemcpy(&tensor_temp, tensors[1], sizeof(tensor3_t), cudaMemcpyDeviceToHost);

	// MaxPool2D
	maxpool2d_k5<<<grid_dim, block_dim>>>(tensors[1], tensors[2]);
	maxpool2d_k5<<<grid_dim, block_dim>>>(tensors[2], tensors[3]);
	maxpool2d_k5<<<grid_dim, block_dim>>>(tensors[3], tensors[4]);

	// Concat
	int block_size = tensor_temp.w * tensor_temp.h * tensor_temp.c;
	tensor_temp.c *= 4;
	cudaMemcpy(tensors[5], &tensor_temp, sizeof(tensor3_t), cudaMemcpyHostToDevice);
	tensor_temp.c /= 4;
	float *out_data = (float*)(tensors[5] + 1);

	for (int i = 0; i < 4; i++) {
		int offset = block_size * i;
		if ((err = cudaMemcpy(out_data + offset, (float*)(tensors[i + 1] + 1), block_size * sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess)
			printf("Process %d: SPPF concat failure for index %d\n\t%s", rank, i, cudaGetErrorString(err));
	}

	// Compute the first second convolution
	grid_dim.z = h_kernels[1]->filters;
	cudaMemset(tensors[0], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(tensors[5], tensors[0], kernels[1]);
}

tensor3_t *detect(tensor3_t *in, int rank) {
	// In case of errors
	cudaError_t err;

	// Send the input tensor to memory
	if ((err = cudaMemcpy(dev_blocks[0], in, sizeof(tensor3_t) + in->w * in->h * in->c * sizeof(float), cudaMemcpyHostToDevice))
		!= cudaSuccess) {
		printf("Process %d: Failed to move image to GPU memory\n\t%s", rank, cudaGetErrorString(err));
		return NULL;
	}

	// Initialize block and grid dim
	dim3 grid_dim(16, 16, h_kernels[0]->filters), block_dim(20, 20);
	cudaMemset(dev_blocks[1], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(dev_blocks[0], dev_blocks[1], dev_kernels[0]);

	grid_dim = {8, 8, (unsigned)h_kernels[1]->filters};
	cudaMemset(dev_blocks[2], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(dev_blocks[1], dev_blocks[2], dev_kernels[1]);

	grid_dim = {8, 8, (unsigned)h_kernels[2]->filters};
	c2f(&dev_blocks[2], &dev_kernels[2], 1, 1, grid_dim, block_dim, rank);

	grid_dim = {4, 4, (unsigned)h_kernels[6]->filters};
	cudaMemset(dev_blocks[0], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(dev_blocks[2], dev_blocks[0], dev_kernels[6]);

	grid_dim = {4, 4, (unsigned)h_kernels[7]->filters};
	c2f(&dev_blocks[0], &dev_kernels[7], 2, 1, grid_dim, block_dim, rank); // The large

	grid_dim = {2, 2, (unsigned)h_kernels[13]->filters};
	cudaMemset(dev_blocks[1], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(dev_blocks[0], dev_blocks[1], dev_kernels[13]);

	grid_dim = {2, 2, (unsigned)h_kernels[14]->filters};
	c2f(&dev_blocks[1], &dev_kernels[14], 2, 1, grid_dim, block_dim, rank); // The medium

	cudaMemset(dev_blocks[2], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	grid_dim = {1, 1, (unsigned)h_kernels[20]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[1], dev_blocks[2], dev_kernels[20]);

	grid_dim = {1, 1, (unsigned)h_kernels[21]->filters};
	c2f(&dev_blocks[2], &dev_kernels[21], 1, 1, grid_dim, block_dim, rank);

	grid_dim = {1, 1, (unsigned)h_kernels[25]->filters};
	sppf(&dev_blocks[2], &dev_kernels[25], &h_kernels[25], grid_dim, block_dim, rank); // The small

	// Aliasing for readability + cache the tensor data
	tensor3_t *neck_feature_80 = dev_blocks[0],
				*neck_feature_40 = dev_blocks[1],
				*neck_feature_20 = dev_blocks[2];
	tensor3_t neck_f80_meta, neck_f40_meta, neck_f20_meta;
	cudaMemcpy(&neck_f80_meta, neck_feature_80, sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&neck_f40_meta, neck_feature_40, sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&neck_f20_meta, neck_feature_20, sizeof(tensor3_t), cudaMemcpyDeviceToHost);

	// Upsample + Concat + C2f
	grid_dim = {2, 2, (unsigned)neck_f20_meta.c};
	cudaMemset(dev_blocks[3], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	upsample<<<grid_dim, block_dim>>>(neck_feature_20, dev_blocks[3]);

	tensor3_t concat_tdl2 = {neck_f40_meta.w, neck_f40_meta.h, neck_f40_meta.c + neck_f20_meta.c};
	cudaMemcpy(dev_blocks[3], &concat_tdl2, sizeof(tensor3_t), cudaMemcpyHostToDevice);
	float *concat_tdl2_cat_point = (float*)(dev_blocks[3] + 1) + neck_f20_meta.c * concat_tdl2.w * concat_tdl2.h;
	int concat_tdl2_amt = neck_f40_meta.w * neck_f40_meta.h * neck_f40_meta.c;
	cudaMemcpy(concat_tdl2_cat_point, (float*)(neck_feature_40 + 1), concat_tdl2_amt * sizeof(float), cudaMemcpyDeviceToDevice);

	grid_dim = {2, 2, (unsigned)h_kernels[27]->filters};
	c2f(&dev_blocks[3], &dev_kernels[27], 1, 0, grid_dim, block_dim, rank);
	tensor3_t *head_intermediary_40 = dev_blocks[3];
	tensor3_t head_med40_meta;
	cudaMemcpy(&head_med40_meta, head_intermediary_40, sizeof(tensor3_t), cudaMemcpyDeviceToHost);

	// Upsample + Concat + C2f
	grid_dim = {4, 4, (unsigned)head_med40_meta.c};
	cudaMemset(dev_blocks[4], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	upsample<<<grid_dim, block_dim>>>(head_intermediary_40, dev_blocks[4]);

	tensor3_t concat_tdl1 = {neck_f80_meta.w, neck_f80_meta.h, neck_f80_meta.c + head_med40_meta.c};
	cudaMemcpy(dev_blocks[4], &concat_tdl1, sizeof(tensor3_t), cudaMemcpyHostToDevice);
	float *concat_tdl1_cat_point = (float*)(dev_blocks[4] + 1) + head_med40_meta.c * concat_tdl1.w * concat_tdl1.h;
	int concat_tdl1_amt = neck_f80_meta.w * neck_f80_meta.h * neck_f80_meta.c;
	cudaMemcpy(concat_tdl1_cat_point, (float*)(neck_feature_80 + 1), concat_tdl1_amt * sizeof(float), cudaMemcpyDeviceToDevice);

	grid_dim = {4, 4, (unsigned)h_kernels[31]->filters};
	c2f(&dev_blocks[4], &dev_kernels[31], 1, 0, grid_dim, block_dim, rank);
	tensor3_t *head_feature_80 = dev_blocks[4];
	tensor3_t head_f80_meta;
	cudaMemcpy(&head_f80_meta, head_feature_80, sizeof(tensor3_t), cudaMemcpyDeviceToHost);

	// Conv 16
	grid_dim = {2, 2, (unsigned)h_kernels[35]->filters};
	cudaMemset(dev_blocks[5], 0, TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t));
	conv<<<grid_dim, block_dim>>>(head_feature_80, dev_blocks[5], dev_kernels[35]);
	err = cudaGetLastError();





	// TODO insert C2f onwards here

    // =========================
    // Bottom-up step 1: 80 -> 40
    // =========================
    // We have:
    //   - dev_blocks[5] : downsampled 80x80 feature (after conv 16)
    //   - head_intermediary_40 : 40x40 feature from the first top-down C2f
    //
    // We want:
    //   concat([conv16_out, head_intermediary_40]) -> C2f(36..) -> head_feature_40

    // Get metadata of conv16 output
    tensor3_t conv16_meta;
    cudaMemcpy(&conv16_meta, dev_blocks[5], sizeof(tensor3_t), cudaMemcpyDeviceToHost);

    // Sanity: conv16 meta should match head_med40_meta.w/h, and channels=h_kernels[35]->filters
    // Build concat tensor metadata: 40x40, (conv16.c + head_med40.c)
    tensor3_t concat_bup0 = {
        head_med40_meta.w,
        head_med40_meta.h,
        conv16_meta.c + head_med40_meta.c
    };

    // Write concat metadata into dev_blocks[5]
    if ((err = cudaMemcpy(dev_blocks[5], &concat_bup0, sizeof(tensor3_t),
                          cudaMemcpyHostToDevice)) != cudaSuccess) {
        printf("Process %d: concat_bup0 meta copy failed\n\t%s\n", rank, cudaGetErrorString(err));
    }

    // Layout:
    //   dev_blocks[5] data = [ conv16_out (first), head_intermediary_40 (second) ]
    //
    // conv16_out is already in dev_blocks[5] data region from the previous conv,
    // so we only need to append head_intermediary_40 after it.

    size_t bup0_spatial = (size_t)concat_bup0.w * concat_bup0.h;
    size_t bup0_conv16_elems = bup0_spatial * (size_t)conv16_meta.c;
    float *concat_bup0_cat_point =
        (float*)(dev_blocks[5] + 1) + bup0_conv16_elems;

    size_t bup0_head40_elems =
        (size_t)head_med40_meta.w * head_med40_meta.h * head_med40_meta.c;

    if ((err = cudaMemcpy(concat_bup0_cat_point,
                          (float*)(head_intermediary_40 + 1),
                          bup0_head40_elems * sizeof(float),
                          cudaMemcpyDeviceToDevice)) != cudaSuccess) {
        printf("Process %d: concat_bup0 data copy failed\n\t%s\n", rank, cudaGetErrorString(err));
    }

    // Run C2f for 40x40 head: uses kernels[36..39]
    grid_dim = {2, 2, (unsigned)h_kernels[36]->filters};
    c2f(&dev_blocks[5], &dev_kernels[36], 1, 0, grid_dim, block_dim, rank);

    tensor3_t *head_feature_40 = dev_blocks[5];
    tensor3_t head_f40_meta;
    cudaMemcpy(&head_f40_meta, head_feature_40, sizeof(tensor3_t), cudaMemcpyDeviceToHost);

    // =========================
    // Bottom-up step 2: 40 -> 20
    // =========================
    // Conv 19: downsample head_feature_40 (stride 2 conv)
    grid_dim = {1, 1, (unsigned)h_kernels[40]->filters};
    conv<<<grid_dim, block_dim>>>(head_feature_40, dev_blocks[6], dev_kernels[40]);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Process %d: Conv 19 failed\n\t%s\n", rank, cudaGetErrorString(err));
    }

    tensor3_t head_f20_down_meta;
    cudaMemcpy(&head_f20_down_meta, dev_blocks[6], sizeof(tensor3_t), cudaMemcpyDeviceToHost);

    // Now concat this downsampled 20x20 head with the original neck_feature_20
    // We already cached neck_f20_meta above.
    tensor3_t concat_bup1 = {
        neck_f20_meta.w,
        neck_f20_meta.h,
        head_f20_down_meta.c + neck_f20_meta.c
    };

    if ((err = cudaMemcpy(dev_blocks[6], &concat_bup1, sizeof(tensor3_t),
                          cudaMemcpyHostToDevice)) != cudaSuccess) {
        printf("Process %d: concat_bup1 meta copy failed\n\t%s\n", rank, cudaGetErrorString(err));
    }

    size_t bup1_spatial = (size_t)concat_bup1.w * concat_bup1.h;
    size_t bup1_head20_elems = bup1_spatial * (size_t)head_f20_down_meta.c;
    float *concat_bup1_cat_point =
        (float*)(dev_blocks[6] + 1) + bup1_head20_elems;

    size_t bup1_neck20_elems =
        (size_t)neck_f20_meta.w * neck_f20_meta.h * neck_f20_meta.c;

    if ((err = cudaMemcpy(concat_bup1_cat_point,
                          (float*)(neck_feature_20 + 1),
                          bup1_neck20_elems * sizeof(float),
                          cudaMemcpyDeviceToDevice)) != cudaSuccess) {
        printf("Process %d: concat_bup1 data copy failed\n\t%s\n", rank, cudaGetErrorString(err));
    }

    // Final C2f for 20x20 head: uses kernels[41..44]
    grid_dim = {1, 1, (unsigned)h_kernels[41]->filters};
    c2f(&dev_blocks[6], &dev_kernels[41], 1, 0, grid_dim, block_dim, rank);

    tensor3_t *head_feature_20 = dev_blocks[6];
    tensor3_t head_f20_meta;
    cudaMemcpy(&head_f20_meta, head_feature_20, sizeof(tensor3_t), cudaMemcpyDeviceToHost);







	// Detect BBoxes
	grid_dim = {4, 4, (unsigned)h_kernels[45]->filters};
	conv<<<grid_dim, block_dim>>>(head_feature_80, dev_blocks[10], dev_kernels[45]);
	grid_dim = {4, 4, (unsigned)h_kernels[46]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[10], dev_blocks[11], dev_kernels[46]);
	grid_dim = {4, 4, (unsigned)h_kernels[47]->filters};
	conv_noswish<<<grid_dim, block_dim>>>(dev_blocks[11], dev_blocks[7], dev_kernels[47]);

	grid_dim = {2, 2, (unsigned)h_kernels[48]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[5], dev_blocks[10], dev_kernels[48]);
	grid_dim = {2, 2, (unsigned)h_kernels[49]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[10], dev_blocks[11], dev_kernels[49]);
	grid_dim = {2, 2, (unsigned)h_kernels[50]->filters};
	conv_noswish<<<grid_dim, block_dim>>>(dev_blocks[11], dev_blocks[8], dev_kernels[50]);

	grid_dim = {2, 2, (unsigned)h_kernels[51]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[6], dev_blocks[10], dev_kernels[51]);
	grid_dim = {2, 2, (unsigned)h_kernels[52]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[10], dev_blocks[11], dev_kernels[52]);
	grid_dim = {2, 2, (unsigned)h_kernels[53]->filters};
	conv_noswish<<<grid_dim, block_dim>>>(dev_blocks[11], dev_blocks[9], dev_kernels[53]);


	// Detect Categories
	grid_dim = {4, 4, (unsigned)h_kernels[54]->filters};
	conv<<<grid_dim, block_dim>>>(head_feature_80, dev_blocks[13], dev_kernels[54]);
	grid_dim = {4, 4, (unsigned)h_kernels[55]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[13], dev_blocks[14], dev_kernels[55]);
	grid_dim = {4, 4, (unsigned)h_kernels[56]->filters};
	conv_noswish<<<grid_dim, block_dim>>>(dev_blocks[14], dev_blocks[10], dev_kernels[56]);

	grid_dim = {4, 4, (unsigned)h_kernels[57]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[5], dev_blocks[13], dev_kernels[57]);
	grid_dim = {4, 4, (unsigned)h_kernels[58]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[13], dev_blocks[14], dev_kernels[58]);
	grid_dim = {4, 4, (unsigned)h_kernels[59]->filters};
	conv_noswish<<<grid_dim, block_dim>>>(dev_blocks[14], dev_blocks[11], dev_kernels[59]);

	grid_dim = {4, 4, (unsigned)h_kernels[60]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[6], dev_blocks[13], dev_kernels[60]);
	grid_dim = {4, 4, (unsigned)h_kernels[61]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[13], dev_blocks[14], dev_kernels[61]);
	grid_dim = {4, 4, (unsigned)h_kernels[62]->filters};
	conv_noswish<<<grid_dim, block_dim>>>(dev_blocks[14], dev_blocks[12], dev_kernels[62]);

	// -------------------------------------
	// 	OUTPUT BBOXES
	// -------------------------------------

	size_t tensor_max_alloc = TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t);
	void *h_blocks_buf = malloc(tensor_max_alloc * PREALLOC_TENSORS);
	tensor3_t *blocks[PREALLOC_TENSORS];
	for (int i = 0; i < PREALLOC_TENSORS; i++) blocks[i] = (tensor3_t*)((char*)h_blocks_buf + tensor_max_alloc * i);
	printf("Process %d: Allocated %d B of memory for %d tensor buffers\n", rank, tensor_max_alloc * PREALLOC_TENSORS, PREALLOC_TENSORS);

	cudaMemcpy(blocks[7], dev_blocks[7], tensor_max_alloc * 6, cudaMemcpyDeviceToHost);



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
			bins[c] = ((float*)(blocks[7] + 1))[
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
			bins[c] = ((float*)(blocks[7] + 1))[
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
			bins[c] = ((float*)(blocks[7] + 1))[
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
			bins[c] = ((float*)(blocks[7] + 1))[
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
			classes[c] = ((float*)(blocks[10] + 1))[
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
				bins[c] = ((float*)(blocks[8] + 1))[
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
				bins[c] = ((float*)(blocks[8] + 1))[
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
				bins[c] = ((float*)(blocks[8] + 1))[
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
				bins[c] = ((float*)(blocks[8] + 1))[
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
				classes[c] = ((float*)(blocks[11] + 1))[
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
					bins[c] = ((float*)(blocks[9] + 1))[
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
					bins[c] = ((float*)(blocks[9] + 1))[
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
					bins[c] = ((float*)(blocks[9] + 1))[
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
					bins[c] = ((float*)(blocks[9] + 1))[
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
					classes[c] = ((float*)(blocks[12] + 1))[
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
		if (boxes[i].class_conf >= conf_thresh) {
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
			if (iou > iou_thresh) {
				boxes[j].inactive = 1;
			}
		}
	}

	// Allocate a tensor to return to the main program
	tensor3_t *ret_val = (tensor3_t*)malloc(sizeof(tensor3_t) + sizeof(int) * 8400 * 6);
	ret_val->w = 6;
	ret_val->h = 8400;


	// Identify # of disabled boxes
	int idx = 0;
	int disabled = 0;
	for (int i = 0; i < total_boxes; ++i) {
		if (boxes[i].inactive) disabled++;
		if (!boxes[i].inactive) {
			void *base_ptr = (void*)((char*)(ret_val + 1) + idx * 6);
			int *classid = (int*)(base_ptr);
			float *box_bounds = (float*)((int*)(base_ptr) + 1);
			classid[0] = boxes[i].cid;
			box_bounds[0] = boxes[i].class_conf;
			box_bounds[1] = boxes[i].x;
			box_bounds[2] = boxes[i].y;
			box_bounds[3] = boxes[i].w;
			box_bounds[4] = boxes[i].h;
			idx++;
		}
	}
	ret_val->c = 8400 - disabled;
	printf("Process %d: Disabled %d boxes\n", rank, disabled);

	free(h_blocks_buf);
	return ret_val;
}

void load_model(conv_t *kernels, int *displacements, size_t block_size, int rank) {
	// In case of errors
	cudaError_t err;

	// Allocate memory for the kernels and copy the kernel data
	if ((err = cudaMalloc(&dev_kernel_buf, block_size)) != cudaSuccess) {
		printf("Process %d: Failed to allocate memory for kernels\n\t%s", rank, cudaGetErrorString(err));
		return;
	}
	if ((err = cudaMemcpy(dev_kernel_buf, (void*)kernels, block_size, cudaMemcpyHostToDevice)) != cudaSuccess) {
		printf("Process %d: Failed to copy kernel memory\n\t%s", rank, cudaGetErrorString(err));
		return;
	}

	// Store the pointers to each kernel
	for (int i = 0; i < NUM_LAYERS; ++i) {
		dev_kernels[i] = (conv_t*)((char*)dev_kernel_buf + displacements[i]);
		h_kernels[i] = (conv_t*)((char*)kernels + displacements[i]);

		/** Used this to print the size per filter for every kernel
		if (rank == 0)
			printf("Kernel found worth %d B per filter as it is %dx%d with %d input channels and %d filters\n",
				   h_kernels[i]->dim * h_kernels[i]->dim * h_kernels[i]->channels,
				   h_kernels[i]->dim, h_kernels[i]->dim, h_kernels[i]->filters, h_kernels[i]->channels);
		*/
	}

	// Allocate memory for the tensors
	size_t tensor_max_alloc = TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t);
	if ((err = cudaMalloc(&dev_blocks_buf, tensor_max_alloc * PREALLOC_TENSORS)) != cudaSuccess) {
		printf("Process %d: Failed to allocate memory for tensor buffers\n\t%s", rank, cudaGetErrorString(err));
		return;
	}
	for (int i = 0; i < PREALLOC_TENSORS; i++) dev_blocks[i] = (tensor3_t*)((char*)dev_blocks_buf + tensor_max_alloc * i);
	printf("Process %d: Allocated %d B of memory for %d tensor buffers\n", rank, tensor_max_alloc * PREALLOC_TENSORS, PREALLOC_TENSORS);

	printf("Process %d: Model loaded and initialized! Consumed %d MiB of global memory\n", rank,
		   (tensor_max_alloc * PREALLOC_TENSORS /*+ im2col_alloc*/ + block_size) / (1 << 20));
}

void free_model(int rank) {
	cudaFree(dev_kernel_buf);
	cudaFree(dev_im2col_buf);
	cudaFree(dev_blocks_buf);
	printf("Process %d: Model has been cleared from memory\n", rank);
}

void device_query(int rank) {
	// Query device properties assuming single device
	cudaDeviceProp prop{};
	cudaError_t err = cudaGetDeviceProperties(&prop, 0);
	if (err != cudaSuccess) {
		printf("No CUDA-capable devices found\n");
		return;
	}

	// Print info
	printf("Process %d Device Information:\n", rank);
	printf("\tDevice Name: %s\n", prop.name);
	printf("\tTotal Global Memory: %u B\n", prop.totalGlobalMem);
	printf("\tMax Threads per Block: %d\n", prop.maxThreadsPerBlock);
	printf("\tMax Shared Mem per Block: %u B\n", prop.sharedMemPerBlock);

	// Store that info
	thds_per_blk = prop.maxThreadsPerBlock;
	shared_mem = prop.sharedMemPerBlock;
}
