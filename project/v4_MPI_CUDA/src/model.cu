#include "model.h"

// Globally accessible device memory pointers
void *dev_kernel_buf; // Raw buffer for alloc and dealloc
void *dev_im2col_buf; // Raw buffer for alloc and dealloc
void *dev_blocks_buf; // Raw buffer for alloc and dealloc
conv_t *dev_kernels[NUM_LAYERS];
tensor3_t *dev_im2col_block;
tensor3_t *dev_blocks[PREALLOC_TENSORS];
conv_t *h_kernels[NUM_LAYERS]; // We need access to the host kernel metadata for im2cole computation

// Global variables
int thds_per_blk;


__global__ void im2col_10(tensor3_t* d_in, tensor3_t* d_out, conv_t* d_kernel, int cc, int hc, int wc) {
	// Compute output coords
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col > hc * wc || row > cc) return;

	const int K = d_kernel->dim;       // ksize
	const int C = d_kernel->channels;  // input channels

	const int H = d_in->h;
	const int W = d_in->w;

	const float* IN  = (float*)(d_in + 1);
	float*       OUT = (float*)(d_out + 1);

	// Decode row -> (c, k) where k is [0 .. K*K-1]
	int kkc = K * K;           // per-channel rows
	int c   = row / kkc;       // which input channel
	int k   = row % kkc;       // which kernel element

	int ky = k / K;            // kernel row
	int kx = k % K;            // kernel col

	// Decode col -> (j, i) = output spatial position
	int j = col / wc;          // 0 .. hc-1 (output y)
	int i = col % wc;          // 0 .. wc-1 (output x)

	// For stride=1, pad=0:
	int in_y = j + ky;
	int in_x = i + kx;

	// Linear index in input (CHW)
	int in_idx = (c * H + in_y) * W + in_x;

	// Linear index in output (row-major)
	int out_idx = row * (hc * wc) + col;

	OUT[out_idx] = IN[in_idx];
}

void im2cole(tensor3_t* d_in, tensor3_t* d_out, conv_t* d_kernel, conv_t* h_kernel) {
	// Retrieve the width and height
	tensor3_t h_in;
	cudaMemcpy(&h_in, d_in, sizeof(tensor3_t), cudaMemcpyDeviceToHost);

	// Compute the output width and height of d_out
	int cc = h_kernel->channels * h_kernel->dim * h_kernel->dim;
	int hc = (h_in.h + 2 * h_kernel->padding - h_kernel->dim) / h_kernel->stride + 1;
	int wc = (h_in.w + 2 * h_kernel->padding - h_kernel->dim) / h_kernel->stride + 1;

	// Retrieve the output tensor
	tensor3_t h_out;
	h_out.w = hc * wc;
	h_out.h = cc;
	h_out.c = 1;
	cudaMemcpy(d_out, &h_out, sizeof(tensor3_t), cudaMemcpyHostToDevice);

	// Compute the block dim and grid dim
	dim3 blk_dim(IM2COL_BLK_DIM, IM2COL_BLK_DIM)
	dim3 grid_dim((h_out.w + blk_dim.x - 1) / blk_dim.x,
				  (h_out.h + blk_dim.y - 1) / blk_dim.y);

	// Launch the appropriate kernel
	if (h_kernel->stride == 1 && h_kernel->padding == 0) {
		im2col_10<<<grid_dim, blk_dim>>>(d_in, d_out, d_kernel, cc, hc, wc);
	} else if (h_kernel->stride == 1 && h_kernel->padding == 1) {

	} else if (h_kernel->stride == 2 && h_kernel->padding == 0) {

	} else if (h_kernel->stride == 2 && h_kernel->padding == 1) {

	}
}

tensor3_t *detect(tensor3_t *in) {

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
	}

	// Allocate memory for the im2col buffer
	size_t im2col_alloc = IM2COL_MAX_SIZE * sizeof(float) + sizeof(tensor3_t);
	if ((err = cudaMalloc(&dev_im2col_buf, im2col_alloc)) != cudaSuccess) {
		printf("Process %d: Failed to allocate memory for im2col buffer\n\t%s", rank, cudaGetErrorString(err));
		return;
	}
	dev_im2col_block = (tensor3_t*)dev_im2col_buf;
	printf("Process %d: Allocated %d B of memory for im2col buffer\n", rank, im2col_alloc);

	// Allocate memory for the im2col buffer
	size_t tensor_max_alloc = TENSOR_MAX_SIZE * sizeof(float) + sizeof(tensor3_t);
	if ((err = cudaMalloc(&dev_blocks_buf, tensor_max_alloc * PREALLOC_TENSORS)) != cudaSuccess) {
		printf("Process %d: Failed to allocate memory for tensor buffers\n\t%s", rank, cudaGetErrorString(err));
		return;
	}
	for (int i = 0; i < PREALLOC_TENSORS; i++) dev_blocks[i] = (tensor3_t*)((char*)dev_blocks_buf + tensor_max_alloc * i);
	printf("Process %d: Allocated %d B of memory for %d tensor buffers\n", rank, tensor_max_alloc * PREALLOC_TENSORS, PREALLOC_TENSORS);

	printf("Process %d: Model loaded and initialized! Consumed %d MiB of global memory\n", rank,
		   (tensor_max_alloc * PREALLOC_TENSORS + im2col_alloc + block_size) / (1 << 20));
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
}
