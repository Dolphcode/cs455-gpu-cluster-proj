#ifndef __MODEL__
#define __MODEL__

#include <iostream>

#include <cuda_runtime.h>

#include "layer_structs.h"

#define		NUM_LAYERS	        65
#define     WEIGHT_BYTES        187376960
#define     PREALLOC_TENSORS    20

#define     KERNEL_MAX_FLOATS   10000 // 2304 floats + bias


#define     TENSOR_MAX_SIZE     (640 * 640 * 32)
#define     IM2COL_MAX_SIZE     (320 * 320 * 16 * 3 * 3)

#define     IM2COL_BLK_DIM      64

#define     DEFAULT_IOU_THRESH  0.3f
#define     DEFAULT_CONF_THRESH 0.25f

extern float iou_thresh;
extern float conf_thresh;

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
tensor3_t* detect(tensor3_t* input, int rank);

#endif
