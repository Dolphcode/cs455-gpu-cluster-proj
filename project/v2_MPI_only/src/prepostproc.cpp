#include <algorithm>

#include <mpi.h>

#include "prepostproc.h"

#define		HALO		320

void mat_to_buf(cv::Mat img, float *buf) {
/*
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
	}*/
}

void load_image_work(std::string filepath, int padding, int rank, int nprocs, tensor3_t **out_buf,int *ntiles) {
	// Compute # tiles
	int tot_tiles = 1, tile_dim = 1;
	while (tot_tiles < nprocs) {
		tot_tiles <<= 2;
		tile_dim <<= 1;
	}

	// Communication prepwork -> every tile will need to be sent but we can kinda work through it asynchronously
	MPI_Request *send_reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * tot_tiles);
	MPI_Request *recv_reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * tot_tiles);

	// Compute my work, but also track all workers so we can keep track of what we're expecting'
	int *work_per_proc = (int*)malloc(sizeof(int) * nprocs);
	for (int i = 0; i < nprocs; i++) {
		work_per_proc[i] = tot_tiles / nprocs;
		if (i < (tot_tiles % nprocs)) work_per_proc[i]++;
	}
	*ntiles = work_per_proc[rank];

	// Compute the size of each tensor
	int tensor_dim = PROCESS_DIM / tile_dim + HALO + 2 * padding; // Tile Size + Halo + Padding
	printf("Process %d is responsible for %d tiles sized %d x %d\n", rank, *ntiles, tensor_dim, tensor_dim);

	// Allocate the buffer into which tensors will be delegated
	size_t size_per_tensor = sizeof(tensor3_t) + tensor_dim * tensor_dim * 3 * sizeof(float);
	*out_buf = (tensor3_t*)malloc(size_per_tensor * (*ntiles));

	if (rank == 0) {
		// Load the image
		printf("Process %d: Loading image at %s with OpenCV\n", rank, filepath.c_str());
		cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);

		// Compute the ratio
		float r = std::min((float)PROCESS_DIM / (float)img.cols, (float)PROCESS_DIM / (float)img.rows);

		// Size down
		printf("Process %d: Image loaded successfully, preprocessing the image\n", rank);
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

		// Prepare the send buffer
		tensor3_t* send_buf = (tensor3_t*)malloc(size_per_tensor * tot_tiles);

		// Send each tile to the corresponding destination
		int curr_proc = 0;
		int curr_proc_tiles = 0;
		int slide_x = (PROCESS_DIM - (tensor_dim - 2 * padding)) / (tile_dim - 1);
		int slide_y = (PROCESS_DIM - (tensor_dim - 2 * padding)) / (tile_dim - 1);
		for (int y = 0; y < tile_dim; y++)
		for (int x = 0; x < tile_dim; x++) {
			// Compute crop bounds
			int x1 = x * slide_x;
			int y1 = y * slide_y;

			// Crop the image
			cv::Mat curr;
			cv::Rect crop_area(x1, y1, tensor_dim - 2 * padding, tensor_dim - 2 * padding);
			curr = resized(crop_area);

			// Select the current buffer
			int curr_tile_index = y * tile_dim + x;
			tensor3_t* curr_buf = (tensor3_t*)((char*)send_buf + size_per_tensor * curr_tile_index);
			curr_buf->data = (float*)(curr_buf + 1);
			curr_buf->w = tensor_dim;
			curr_buf->h = tensor_dim;
			curr_buf->c = 3;

			// Load the cropped image into the tensor
			for (int c = 0; c < 3; ++c) {
				for (int row = padding; row < tensor_dim - padding; row++) {
					for (int col = padding; col < tensor_dim - padding; col++) {
						// Compute the index in the raw image
						int im_idx = (row - padding) * (tensor_dim - 2 * padding) * curr_buf->c +
						(col - padding) * curr_buf->c +
						(2 - c);

						// Compute the index in the buffer
						// Organized like this to make splitting easier
						int buf_idx = c * curr_buf->w * curr_buf->h +
						row * curr_buf->w +
						col;

						// Transfer to the buffer. Normalize the colors
						curr_buf->data[buf_idx] = curr.data[im_idx] / 255.0;
					}
				}
			}

			// Send data to the appropriate place
			if (curr_proc == 0) {
				// Just copy into the out buffer
				tensor3_t* copy_loc = (tensor3_t*)((char*)(*out_buf) + curr_proc_tiles * size_per_tensor);
				memcpy((void*)copy_loc, (void*)curr_buf, size_per_tensor);
				printf("Process %d will work on %dx%d tensor with %d channels\n", rank,
					   ((tensor3_t*)copy_loc)->w,
					   ((tensor3_t*)copy_loc)->h,
					   ((tensor3_t*)copy_loc)->c);

				copy_loc->data = (float*)(copy_loc + 1);
			} else {
				// Immediate send to other processes
				MPI_Isend(
					(void*)curr_buf,
					size_per_tensor,
					MPI_BYTE,
					curr_proc,
					curr_proc_tiles,
					MPI_COMM_WORLD,
					send_reqs + curr_tile_index
				);
			}

			// Shift the process that data is going to
			curr_proc_tiles++;
			if (curr_proc_tiles >= work_per_proc[curr_proc]) {
				curr_proc_tiles = 0;
				curr_proc++;
			}
		}

		// TODO MPI_Wait
		for (int i = *ntiles; i < tot_tiles; ++i)
			MPI_Wait(send_reqs + i, MPI_STATUS_IGNORE);

		// Clear once we know everything has been sent
		free(send_buf);
	} else {
		// Determine where the requests I want to wait for reside
		int req_idx = 0;
		for (int i = 0; i < rank; ++i) req_idx += work_per_proc[i];

		// Receive
		for (int i = 0; i < *ntiles; i++) {
			tensor3_t* curr_buf = (tensor3_t*)((char*)(*out_buf) + size_per_tensor * i);
				MPI_Irecv((void*)curr_buf,
				size_per_tensor,
				MPI_BYTE,
				0,
				i,
				MPI_COMM_WORLD,
				recv_reqs + i + req_idx);
		}

		// Await
		for (int i = 0; i < *ntiles; i++) {
			MPI_Wait(recv_reqs + req_idx + i, MPI_STATUS_IGNORE);
			tensor3_t* received = (tensor3_t*)((char*)(*out_buf) + size_per_tensor * i);
			printf("Process %d received tensor %dx%d with %d channels\n",
					rank,
					received->w,
					received->h,
					received->c
			);
			received->data = (float*)(received + 1); // Make sure data is pointing to the correct spot
		}
	}

	free(send_reqs);
	free(recv_reqs);
	free(work_per_proc);
}





tensor3_t* load_image(std::string filepath, int padding, void *buf, int rank, int nprocs) {
	// Verify that buf is real
	if (!buf) return NULL;

	// Some buffers for distribution
	float *tile_buf;
	int *images;

	// Compute the number of tiles we need
	int tiles = 1;
	while (tiles < nprocs) tiles <<= 1;

	// Master will load the image and perform cropping and everything
	if (rank == 0) {
		// Load the image
		printf("Process %d: Loading image at %s with OpenCV\n", rank, filepath.c_str());
		cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);

		// Compute the ratio
		float r = std::min((float)PROCESS_DIM / (float)img.cols, (float)PROCESS_DIM / (float)img.rows);

		// Size down
		printf("Process %d: Image loaded successfully, preprocessing the image\n", rank);
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
	}

	// Return the original pointer as a tensor3_t pointer essentially
	return NULL;
}
