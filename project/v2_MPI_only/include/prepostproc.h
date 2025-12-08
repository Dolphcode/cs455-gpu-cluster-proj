#ifndef __PRE_POST_PROCESS__
#define __PRE_POST_PROCESS__

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <mpi.h>

#include "layer_structs.h"

#define PROCESS_DIM 1280

/**
 * This header file contains the function for loading and preprocessing
 * an image for the forward pass using OpenCV, and the function for saving and
 * outputting an image with bounding boxes
 */

/**
 * Reads an image, preprocesses it, and loads the raw bytes to a buffer whilst
 * also providing shape information via a tensor3_t struct
 */
tensor3_t* load_image(
		std::string filepath,
		int padding,
		void *buf,
		int rank,
		int nprocs
);

void load_image_work(
	std::string filepath,
	int padding,
	int rank,
	int nprocs,
	tensor3_t **out_buf,
	int *ntiles
);

#endif
