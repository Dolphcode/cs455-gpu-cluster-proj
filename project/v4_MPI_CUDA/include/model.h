#include <iostream>

#include <cuda_runtime.h>

#include "layer_structs.h"

#define		NUM_LAYERS	        65
#define     WEIGHT_BYTES        18737696
#define     PREALLOC_TENSORS    15
#define     TENSOR_MAX_SIZE     (320 * 320 * 16)
#define     IM2COL_MAX_SIZE     (320 * 320 * 16 * 3 * 3)

#define     IM2COL_BLK_DIM     64

/**
 * Prints some information about the device
 */
void device_query(int);

/**
 * Loads the model to the GPU
 */
void load_model(conv_t *kernels, int *displacements, size_t block_size, int rank);

/**
 * Frees the model weights from the GPU
 */
void free_model(int rank);

/**
 * Executes the model and returns a buffer for the output
 */
tensor3_t* detect(tensor3_t* input);

