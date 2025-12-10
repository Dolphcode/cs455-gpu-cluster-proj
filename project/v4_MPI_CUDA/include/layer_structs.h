#ifndef __LAYER_STRUCTS__
#define __LAYER_STRUCTS__

#include <cstdio>

typedef struct {
    int x;
    int y;
    int w;
    int h;
    int cid;
    float class_conf;
    int inactive;
}bbox_t;


/**
 * A struct defining metadata for a convolutional layer assuming a square kernel
 * The actual kernel data will be parsed under the assumption that kernel points
 * to a contiguous block of memory containing an alternating sequence of flattened
 * kernels and their corresponding biases:
 * 	filter0, bias0, filter1, bias1, ...
 */
typedef struct {
    int	dim;
    int	channels;
    int	stride;
    int	padding;
    int	filters;
    int	data_len;
}conv_t;

/**
 * Computes the amount of space to allocate for a convolutional layer and returns
 * the result.
 */
size_t conv_malloc_amt(
    int dim,
    int channels,
    int filters
);

/**
 * Computes the amount of space to allocate for a c2f layer and returns
 * the result.
 */
size_t c2f_malloc_amt(
    int in_channels,
    int out_channels,
    int n
);

/**
 * Computes the amount of space to allocate for a SPPF layer and returns
 * the result.
 */
size_t sppf_malloc_amt(
    int in_channels
);

/**
 * Reads a single convolutional layer from a file
 */
void* fread_conv(FILE *infile, void *buf);

/**
 * A 3D tensor type struct containing all the metadata to be able to parse the
 * tensor
 */
typedef struct {
    int w;
    int h;
    int c;
}tensor3_t;

#endif
