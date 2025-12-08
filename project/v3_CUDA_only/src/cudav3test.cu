#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

#include "layer_structs.h"
#include "cuda_layers.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Kernel that mirrors conv_layer_serial's inner loops
__global__ void conv2d_sigmoid_kernel(
    const float* __restrict__ in_data,
    float*       __restrict__ out_data,
    const float* __restrict__ kdata,
    int in_w, int in_h, int in_c,
    int out_w, int out_h,
    int dim, int k_channels, int filters,
    int stride, int k_padding,
    int padding
)
{
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int filter  = blockIdx.z * blockDim.z + threadIdx.z;

    if (filter >= filters || out_row >= out_h || out_col >= out_w) return;

    if (out_row < padding || out_row >= (out_h - padding) ||
        out_col < padding || out_col >= (out_w - padding)) {
        return;
    }

    int x_pix = ((out_col - padding) * stride) + k_padding;
    int y_pix = ((out_row - padding) * stride) + k_padding;

    int weights_per_filter = dim * dim * k_channels;
    int filter_offset      = filter * (weights_per_filter + 1); // +1 for bias
    const float* curr_kernel = &kdata[filter_offset];

    float sum = 0.0f;

    for (int z = 0; z < k_channels; z++) {
        for (int y = 0; y < dim; y++) {
            for (int x = 0; x < dim; x++) {
                int kernel_x_offset = x - (dim / 2);
                int kernel_y_offset = y - (dim / 2);

                int in_x = x_pix + kernel_x_offset;
                int in_y = y_pix + kernel_y_offset;

                float in_pix = in_data[
                    z * (in_w * in_h) +
                    in_y * in_w +
                    in_x
                ];

                float w = curr_kernel[
                    z * dim * dim +
                    y * dim +
                    x
                ];

                sum += in_pix * w;
            }
        }
    }

    // Bias: same index as CPU version
    float bias = kdata[
        (weights_per_filter + 1) * filter +
        weights_per_filter
    ];
    sum += bias;

    // Sigmoid and output: out = sum * sigmoid(sum)
    float sigmoid = 1.0f / (1.0f + expf(-sum));

    int out_idx =
        filter * (out_w * out_h) +
        out_row * out_w +
        out_col;

    out_data[out_idx] = sum * sigmoid;
}

void conv_layer_cuda(tensor3_t *in, tensor3_t *out, conv_t *kernel, int padding, int keep_tensor)
{
    if (!in || !out || !kernel || !in->data || !kernel->kernel) {
        fprintf(stderr, "conv_layer_cuda: null input/output/kernel\n");
        return;
    }

    // Match the CPU's output shape logic if keep_tensor == 0
    if (!keep_tensor) {
        out->w = 2 * padding + ((in->w - 2 * kernel->padding) / kernel->stride);
        out->h = 2 * padding + ((in->h - 2 * kernel->padding) / kernel->stride);
        out->c = kernel->filters;
        out->data = (float*)(out + 1);
    }

    int in_w = in->w;
    int in_h = in->h;
    int in_c = kernel->channels;   // CPU uses kernel->channels in the loops

    int out_w = out->w;
    int out_h = out->h;
    int filters = kernel->filters;

    size_t in_elems  = (size_t)in_w * in_h * in_c;
    size_t out_elems = (size_t)out_w * out_h * filters;
    size_t k_bytes   = (size_t)kernel->data_len * sizeof(float);

    float *d_in  = nullptr;
    float *d_out = nullptr;
    float *d_k   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  in_elems  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k,   k_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, in->data,
                          in_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, kernel->kernel,
                          k_bytes,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(float)));

    dim3 block(16, 16, 1);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        filters
    );

    conv2d_sigmoid_kernel<<<grid, block>>>(
        d_in,
        d_out,
        d_k,
        in_w, in_h, in_c,
        out_w, out_h,
        kernel->dim,
        kernel->channels,
        kernel->filters,
        kernel->stride,
        kernel->padding,
        padding
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out->data, d_out,
                          out_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_k));
}

