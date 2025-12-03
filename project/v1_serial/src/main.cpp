#include <iostream>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "prepostproc.h"
#include "layer_structs.h"

#define 	MAX_TENSOR_BLOCK	(320 * 320 * 16 + 8)

int main(int argc, char *argv[]) {
	// Allocate a big buffer
	void *img_buf = calloc(MAX_TENSOR_BLOCK, sizeof(float));

	tensor3_t* img_tensor = load_image("./deer.jpg", 1, img_buf);
	printf("Image successfully loaded!\n");

	printf("Pixel 1,1\n(%f, %f, %f)\n",
			img_tensor->data[640 + 1] * 255.0,
			img_tensor->data[640 * 640 + 640 + 1] * 255.0,
			img_tensor->data[2 * 640 * 640 + 640 + 1] * 255.0
	      );

	// Allocate 
	void *conv_buf = calloc(conv_malloc_amt(3, 3, 16) / sizeof(float),
			sizeof(float));

	conv_t* conv = (conv_t*)conv_buf;
	
	FILE *infile = fopen("./filters.bin", "rb");
	fread_conv(infile, conv_buf);

	printf("Loaded the kernel!\n");
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				printf("%f\t", conv->kernel[9 * i + 3 * j + k]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("Bias: %f\n", conv->kernel[27]);
	

	
	fclose(infile);
	free(conv_buf);
	free(img_buf);
}
