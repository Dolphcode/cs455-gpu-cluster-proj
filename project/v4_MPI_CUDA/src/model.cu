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

__global__ void sum(float *in, float *out, int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
		out[index] += in[index];
}

__global__ void conv(tensor3_t *in, tensor3_t *out, conv_t *kernel) {
	// Data Pointers
	float *kernel_data = (float*)(kernel + 1);
	float *in_data = (float*)(in + 1);
	float *out_data = (float*)(out + 1);

	// Store kernel and input tensor metadata in shared memory
	__shared__ tensor3_t in_meta;
	__shared__ conv_t kernel_meta;

	// Load to shared memory
	__shared__ float block_filter[KERNEL_MAX_FLOATS];
	int filter = blockIdx.z;
	int tid = blockDim.x * threadIdx.y + threadIdx.x;

	// Cache input data and
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

	// Compute swish
	float sigmoid = 1.0f / (1.0f + __expf(-sum));

	// Set the output
	out_data[out_index] = sigmoid * sum;
}

void c2f(tensor3_t **tensors, conv_t** kernels, short n, short shortcut, dim3 grid_dim, dim3 block_dim, int rank) {
	// In case of errors
	cudaError_t err;

	// Compute the first convolution and extract the output tensor
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

	/*
	// Bottleneck
	grid_dim.z = c2f_out.c;
	conv<<<grid_dim, block_dim>>>(tensors[2], tensors[4], kernels[1]);
	conv<<<grid_dim, block_dim>>>(tensors[4], tensors[3], kernels[2]);
	int thread_count = c2f_out.c * c2f_out.w * c2f_out.h;
	sum<<<thread_count / 256, 256>>>((float*)(tensors[2] + 1), (float*)(tensors[3] + 1));

	// Concat
	c2f_out.c *= (n + 2);
	if ((err = cudaMemcpy(tensors[4], &c2f_out, sizeof(tensor3_t), cudaMemcpyHostToDevice)) != cudaSuccess)
		printf("Process %d: Failed to setup concat tensor\n\t%s", rank, cudaGetErrorString(err));

	for (int j = 0; j < n + 2; j++) {
		// Compute the output block offset
		int concat_offset = j * split_block_size;
		float* out_block = (float*)(tensors[4] + 1) + concat_offset;

		// Copy each block to the concat tensor
		if ((err = cudaMemcpy(out_block, (float*)(tensors[1 + j] + 1), sizeof(float) * split_block_size, cudaMemcpyDeviceToDevice)) != cudaSuccess)
			printf("Process %d: Failed copy block %d to concat tensor\n\t%s", rank, j, cudaGetErrorString(err));
	}

	// Final conv -> Output into tensors 0
	grid_dim.z = out_channels;
	conv<<<grid_dim, block_dim>>>(tensors[4], tensors[0], kernels[3]);*/

	// Bottleneck
	grid_dim.z = c2f_out.c;
	for (int i = 0; i < n; i++) {
		conv<<<grid_dim, block_dim>>>(tensors[2 + i], tensors[4 + i], kernels[1 + i * 2]);
		conv<<<grid_dim, block_dim>>>(tensors[4 + i], tensors[3 + i], kernels[2 + i * 2]);

		int thread_count = c2f_out.c * c2f_out.w * c2f_out.h;
		sum<<<thread_count / 256, 256>>>((float*)(tensors[2 + i] + 1), (float*)(tensors[3 + i] + 1), thread_count);
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
	conv<<<grid_dim, block_dim>>>(tensors[3 + n], tensors[0], kernels[1 + 2 * n]);

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

	// Conv idx 0
	conv<<<grid_dim, block_dim>>>(dev_blocks[0], dev_blocks[1], dev_kernels[0]);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Process %d: Error!\n\t%s\n", rank, cudaGetErrorString(err));
	}

	float value = 0;
	tensor3_t test;
	cudaMemcpy(&value, dev_blocks[1] + 1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&test, dev_blocks[1], sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);
	printf("Process %d: c2f complete %d %d %d\n", rank, test.w, test.h, test.c);
	value = 0;
	cudaMemcpy(&value, (float*)(dev_blocks[1] + 1) + 1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);

	// Conv idx 1
	grid_dim = {8, 8, (unsigned)h_kernels[1]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[1], dev_blocks[2], dev_kernels[1]);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Process %d: Error!\n\t%s\n", rank, cudaGetErrorString(err));
	}

	value = 0;
	cudaMemcpy(&value, dev_blocks[2] + 1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&test, dev_blocks[2], sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	printf("Process %d: Convolution complete %f\n", rank, value);
	printf("Process %d: Convolution complete %d %d %d\n", rank, test.w, test.h, test.c);
	value = 0;
	cudaMemcpy(&value, (float*)(dev_blocks[2] + 1) + 1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);

	// C2f
	c2f(dev_blocks + 2, dev_kernels + 2, 1, 1, grid_dim, block_dim, rank);
	if (err != cudaSuccess) {
		printf("Process %d: Error!\n\t%s\n", rank, cudaGetErrorString(err));
	}

	value = 0;
	cudaMemcpy(&value, dev_blocks[2] + 1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&test, dev_blocks[2], sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);
	printf("Process %d: c2f complete %d %d %d\n", rank, test.w, test.h, test.c);
	value = 0;
	cudaMemcpy(&value, (float*)(dev_blocks[2] + 1) + 1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);

	// Conv idx 3
	grid_dim = {4, 4, (unsigned)h_kernels[6]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[2], dev_blocks[3], dev_kernels[6]);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Process %d: Error!\n\t%s\n", rank, cudaGetErrorString(err));
	}

	value = 0;
	cudaMemcpy(&value, dev_blocks[3] + 1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&test, dev_blocks[3], sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	printf("Process %d: Convolution complete %f\n", rank, value);
	printf("Process %d: Convolution complete %d %d %d\n", rank, test.w, test.h, test.c);
	value = 0;
	cudaMemcpy(&value, (float*)(dev_blocks[3] + 1) + 1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Process %d: Convolution complete %f\n", rank, value);

	// C2f
	c2f(dev_blocks + 3, dev_kernels + 7, 2, 1, grid_dim, block_dim, rank);
	if (err != cudaSuccess) {
		printf("Process %d: Error!\n\t%s\n", rank, cudaGetErrorString(err));
	}

	value = 0;
	cudaMemcpy(&value, dev_blocks[3] + 1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&test, dev_blocks[3], sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);
	printf("Process %d: c2f complete %d %d %d\n", rank, test.w, test.h, test.c);
	value = 0;
	cudaMemcpy(&value, (float*)(dev_blocks[3] + 1) + 1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);

	// Conv idx 5
	grid_dim = {2, 2, (unsigned)h_kernels[13]->filters};
	conv<<<grid_dim, block_dim>>>(dev_blocks[3], dev_blocks[4], dev_kernels[13]);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Process %d: Error!\n\t%s\n", rank, cudaGetErrorString(err));
	}

	value = 0;

	cudaMemcpy(&value, dev_blocks[4] + 1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&test, dev_blocks[4], sizeof(tensor3_t), cudaMemcpyDeviceToHost);
	printf("Process %d: Convolution complete %f\n", rank, value);
	printf("Process %d: Convolution complete kernel value is %f\n", rank, *(float*)(h_kernels[20] + 1));
	printf("Process %d: Convolution complete %d %d %d\n", rank, test.w, test.h, test.c);
	value = 0;
	cudaMemcpy(&value, (float*)(dev_blocks[4] + 1) + 1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Process %d: Convolution complete %f\n", rank, value);
	printf("Testing Kernel, %d %d %d %d\n", h_kernels[20]->dim, h_kernels[20]->filters, h_kernels[20]->channels, h_kernels[20]->padding);

	// C2f
	c2f(dev_blocks + 4, dev_kernels + 14, 2, 1, grid_dim, block_dim, rank);
	if (err != cudaSuccess) {
		printf("Process %d: Error!\n\t%s\n", rank, cudaGetErrorString(err));
	}

	value = 0;
	cudaMemcpy(&value, dev_blocks[4] + 1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Process %d: c2f complete %f\n", rank, value);
	printf("Testing Kernel, %d %d %d %d\n", h_kernels[14]->dim, h_kernels[14]->filters, h_kernels[14]->channels, h_kernels[14]->padding);


	return NULL;
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

	/*
	// Allocate memory for the im2col buffer
	size_t im2col_alloc = IM2COL_MAX_SIZE * sizeof(float) + sizeof(tensor3_t);
	if ((err = cudaMalloc(&dev_im2col_buf, im2col_alloc)) != cudaSuccess) {
		printf("Process %d: Failed to allocate memory for im2col buffer\n\t%s", rank, cudaGetErrorString(err));
		return;
	}
	dev_im2col_block = (tensor3_t*)dev_im2col_buf;
	printf("Process %d: Allocated %d B of memory for im2col buffer\n", rank, im2col_alloc);
	*/

	// Allocate memory for the im2col buffer
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
