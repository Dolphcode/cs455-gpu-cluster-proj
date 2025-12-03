#include <iostream>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "prepostproc.h"
#include "layer_structs.h"

#define 	MAX_TENSOR_BLOCK	(320 * 320 * 16 + 8)

int main(int argc, char *argv[]) {
	// Allocate a big buffer
	void *buf = calloc(MAX_TENSOR_BLOCK, sizeof(float));

	tensor3_t* test = load_image("./deer.jpg", 1, buf);
	printf("Image successfully loaded!\n");

	printf("Pixel 1,1\n(%f, %f, %f)\n",
			test->data[640 + 1] * 255.0,
			test->data[640 * 640 + 640 + 1] * 255.0,
			test->data[2 * 640 * 640 + 640 + 1] * 255.0
	      );

	free(buf);
}
