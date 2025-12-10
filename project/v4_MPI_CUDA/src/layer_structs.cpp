#include "layer_structs.h"

size_t conv_malloc_amt(int dim, int channels, int filters) {
    // Initialize to 0
    size_t alloc_amt = 0;

    // Space for the convolution metadata
    alloc_amt += sizeof(conv_t);

    // Space for the filters themselves
    alloc_amt += (dim * dim * channels * filters) * sizeof(float);

    // Bias terms
    alloc_amt += filters * sizeof(float);

    return alloc_amt;
}

size_t c2f_malloc_amt(int in_channels, int out_channels, int n) {
    size_t alloc_amt = 0;

    // The first conv layer
    alloc_amt += conv_malloc_amt(1, in_channels, out_channels);

    // The bottlenecks
    alloc_amt += 2 * n * conv_malloc_amt(3, out_channels / 2, out_channels / 2);

    // The last conv layer
    alloc_amt += conv_malloc_amt(1, (out_channels / 2) * (n + 2), out_channels);

    return alloc_amt;
}

size_t sppf_malloc_amt(int in_channels) {
    size_t alloc_amt = 0;

    alloc_amt += 2 * conv_malloc_amt(1, in_channels, in_channels);

    return alloc_amt;
}

void* fread_conv(FILE *infile, void *buf) {
    // Validate the file and the buffer
    if (!infile || !buf) return NULL;

    // Read from the file into buffer
    fgetc(infile);
    if (feof(infile)) {
        printf("EOF Reached\n");
        return NULL;
    }
    fseek(infile, -1, SEEK_CUR);

    // Read from the file into the buffer
    fread(buf, sizeof(int), 5, infile);

    /* DEBUG printing file reading
     *	printf("Read Conv: %d %d %d %d %d at %x\n", *((int*)buf),
     *((int*)buf + 1),
     *((int*)buf + 2),
     *((int*)buf + 3),
     *((int*)buf + 4),
     *			ftell(infile));
     */

    // Convert the buf pointer to a convolution pointer
    conv_t *metadata = (conv_t*)buf;

    // Compute the data length
    metadata->data_len = (metadata->dim * metadata->dim * metadata->channels + 1) * metadata->filters;

    // Set the data pointer
    float *kernel = (float*)(metadata + 1);

    // Read into the buffer
    fread(kernel, sizeof(float), metadata->data_len, infile);

    // Return the pointer after
    return (void*)(kernel + metadata->data_len);
}
