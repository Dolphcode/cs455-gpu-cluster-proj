#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "layer_structs.h"
#include "serial_funcs.h"


int main(int argc, char *argv[]) {
	// Load image
	printf("Loading image with OpenCV\n");
	cv::Mat img = cv::imread("./deer.jpg", cv::IMREAD_COLOR);
	cv::Size newsize(640, 640);
	cv::Mat resized;
	printf("Preprocessing image\n");
	cv::resize(img, resized, newsize);
	printf("Decoding image\n");

	// Compute the kernel block size
	// for testing let's just use the first two conv layers of YOLOv8
	size_t block_size = 0, data_size = 0;
	conv_malloc_amt(3, 3, 16, 1, 640, 640, 1, &block_size, &data_size); // Conv 1

	printf("For just conv1 %dB, %dB to allocate\n", block_size, data_size);
	
	conv_malloc_amt(3, 16, 32, 1, 320, 320, 1, &block_size, &data_size); // Conv 2

	printf("%dB, %dB to allocate\n", block_size, data_size);

	void *conv_ptr = calloc(block_size / sizeof(float), sizeof(float));
	void *data_ptr = calloc(data_size / sizeof(float), sizeof(float));
	void *curr = conv_ptr;

	curr = conv_layer(3, 3, 2, 1, 16, curr);
	
	// Load kernel from file
	FILE *f = fopen("./kern.bin", "rb");
	conv_t *k = ((conv_t*)conv_ptr);
	read_conv(f, k);
	fclose(f);

	curr = conv_layer(3, 16, 2, 1, 32, curr);

	// Now construct the input tensor and the output tensor
	tensor_t *img_bytes = (tensor_t*)data_ptr;
	tensor_t *next = (tensor_t*)create_tensor(642, 642, 3, 1, data_ptr);

	float* test = (float*)create_tensor(322, 322, 16, 1, (void*)next);

	unsigned *image_data_ptr = (unsigned*)resized.data;
	for (int i = 0; i < 640; i++) {
		for (int j = 0; j < 640; j++) {
			for (int c = 0; c < 3; ++c) {
				int idx = (i + 1) * 642 * 3 + (j + 1) * 3 + c;
				img_bytes->data[idx] =  resized.data[i * 640 * 3 + j * 3 + c] / 255.0;
				

			}
		}
	}

	printf("Finished loading raw bytes %f %f\n",
			img_bytes->data[0], 
			img_bytes->data[642 * 3 + 3 + 2]
			);

	// Computing convolution
	conv_2d_serial_forward(img_bytes, next, k);
	
	printf("Test: %f\n", next->data[322 * 16 * 20 + 16 * 20 + 3]);
	free(conv_ptr);
	free(data_ptr);
	return 0;
}
