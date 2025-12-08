#include <algorithm>

#include "prepostproc.h"

tensor3_t* load_image(std::string filepath, int padding, void *buf) {
	// Verify that buf is real
	if (!buf) return NULL;

	// Load the image
	printf("Loading image at %s with OpenCV\n", filepath.c_str());
	cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);

	// Compute the ratio
	float r = std::min((float)PROCESS_DIM / (float)img.cols, (float)PROCESS_DIM / (float)img.rows);

	// Size down
	printf("Image loaded successfully, preprocessing the image\n");
	cv::Size newsize((int)(std::round(img.cols * r)), (int)(std::round(img.rows * r)));
	cv::Mat resized_down;
	cv::resize(img, resized_down, newsize, 0, 0, cv::INTER_LINEAR);

	int pad_w = PROCESS_DIM - newsize.width;
	int pad_h = PROCESS_DIM - newsize.height;
	int left = pad_w / 2;
	int right = pad_w - left;
	int top = pad_h / 2;
	int bottom = pad_h - top;


	// Pad the image
	cv::Mat resized;
	cv::copyMakeBorder(resized_down, resized,
					   top, bottom, left, right,
					   cv::BORDER_CONSTANT,
					   cv::Scalar(114, 114, 114));

	// Prepare the tensor
	tensor3_t *metadata = (tensor3_t*)buf;
	metadata->c = 3;
	metadata->w = PROCESS_DIM + 2 * padding;
	metadata->h = PROCESS_DIM + 2 * padding;
	metadata->data = (float*)(metadata + 1);

	// Populate the tensor
	// TODO: May need to reverse the channel loop
	for (int c = 0; c < 3; ++c) {
		for (int row = padding; row < padding + PROCESS_DIM; row++) {
			for (int col = padding; col < padding + PROCESS_DIM; col++) {
				// Compute the index in the raw image
				int im_idx = (row - padding) * PROCESS_DIM * metadata->c +
					(col - padding) * metadata->c +
					(2 - c);

				// Compute the index in the buffer
				// Organized like this to make splitting easier
				int buf_idx = c * metadata->w * metadata->h +
					row * metadata->w +
					col;
				
				// Transfer to the buffer. Normalize the colors
				metadata->data[buf_idx] = resized.data[im_idx] / 255.0;
			}
		}
	}
		
	// Return the original pointer as a tensor3_t pointer essentially
	return metadata;
}
