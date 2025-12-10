#include <iostream>

#include <opencv2/opencv.hpp>

#include <mpi.h>

#include "layer_structs.h"
#include "model.h"
#include "preprocess.h"

#define		ROOT		0

#define		MAX_PROCS	16

#define 	SEND_FRAME		123
#define		RECV_FRAME		6769

size_t compute_conv_alloc_amt();

int main(int argc, char *argv[]) {
	// MPI Initialization
	int my_rank, nprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Print device information
	device_query(my_rank);

	// Load the weights first on the root device, then send to each of the other devices
	size_t conv_block_size = compute_conv_alloc_amt();
	unsigned char *conv_buf = (unsigned char*)calloc(sizeof(char), conv_block_size); // Store the addresses of conv layers
	unsigned char *curr = conv_buf;
	int conv_layer_disps[NUM_LAYERS], curr_kernel = 0; // We will compute the byte differences of addresses

	if (my_rank == ROOT) {
		FILE *infile = fopen("filters.bin", "rb");
		while (1) {
			// Exit out if we hit the EOF
			if (curr == NULL) break;

			// Compute the displacement
			conv_layer_disps[curr_kernel++] = curr - conv_buf;

			// Save the kernel pointer, then read the next kernel
			curr = (unsigned char*)fread_conv(infile, (conv_t*)curr);
		}
		fclose(infile);
		printf("Process %d: Read all %d convolution layers, total %d B\n", my_rank, curr_kernel, conv_block_size);

		// Send
		for (int i = 1; i < nprocs; i++) {
			MPI_Send(conv_layer_disps, NUM_LAYERS, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(conv_buf, conv_block_size, MPI_BYTE, i, 1, MPI_COMM_WORLD);
		}

	} else {
		// Receive per rank
		MPI_Recv(conv_layer_disps, NUM_LAYERS, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(conv_buf, conv_block_size, MPI_BYTE, ROOT, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	// Move kernel data to the GPU
	load_model((conv_t*)conv_buf, conv_layer_disps, conv_block_size, my_rank);

	// Load parameters from argc argv
	std::string filepath(argv[1]);
	if (argc > 2) iou_thresh = atof(argv[2]);
	if (argc > 3) conf_thresh = atof(argv[3]);

	// Create a buffer for the input tensor
	void *im_buf = malloc(sizeof(tensor3_t) + sizeof(float) * PROCESS_DIM * PROCESS_DIM * 3);
	void *send_buffer;
	tensor3_t* out_tensor;

	if (my_rank == ROOT) {
		// Get the video and relevant information
		cv::VideoCapture cap(filepath);
		if (!cap.isOpened()) {
			printf("ROOT: Process not found\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		double fps      = cap.get(cv::CAP_PROP_FPS);
		int    width    = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
		int    height   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
		int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		cv::Size frame_size(width, height);


		// Initialize the writer
		cv::VideoWriter writer(
			"./output.mp4",
			fourcc,         // codec
			fps > 0 ? fps : 30.0, // fallback FPS
			frame_size,
			true            // isColor
		);

		// Prepare the list of frames to buffer and the workers buffers for staging to send
		cv::Mat frames_buffered[MAX_PROCS];
		size_t send_size = (sizeof(tensor3_t) + sizeof(float) * PROCESS_DIM * PROCESS_DIM * 3);
		send_buffer = malloc(send_size * (nprocs - 1));
		char *worker_buffers[MAX_PROCS];
		for (int i = 0; i < nprocs - 1; i++)
			worker_buffers[i + 1] = (char*)send_buffer + (i * send_size);
		worker_buffers[0] = (char*)im_buf;

		// Prepare the requests and the buffer to store the worker output
		MPI_Request requests[MAX_PROCS];
		size_t output_max_size = sizeof(tensor3_t) + sizeof(int) * 8400 * 6;
		void *out_buf = malloc(output_max_size);

		int frames_parsed = 0;
		// Parsing loop
		while (true) {
			// Buffer frames for each process
			int nframes = 0;
			for (int i = 0; i < nprocs; i++) {
				if (!cap.read(frames_buffered[i])) break;
				nframes++;
			}
			if (nframes == 0) {
				// Send a terminator message
				for (int i = 1; i < nprocs; i++) {
					tensor3_t *metadata = (tensor3_t*)(worker_buffers[i]);
					metadata->c = -1;
					MPI_Isend(worker_buffers[i], send_size, MPI_BYTE, i, SEND_FRAME, MPI_COMM_WORLD, &requests[i]);
					MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
				}
				break;
			}

			// Load and send each image
			for (int i = 0; i < nframes; i++) {
				// Get the buffered frame
				cv::Mat img = frames_buffered[i];

				// Compute the ratio
				float r = std::min((float)PROCESS_DIM / (float)img.cols, (float)PROCESS_DIM / (float)img.rows);

				// Size down
				cv::Size newsize((int)(std::round(img.cols * r)), (int)(std::round(img.rows * r)));
				cv::Mat resized_down;
				cv::resize(img, resized_down, newsize, 0, 0, cv::INTER_LINEAR);

				// Pad the image
				int pad_w = PROCESS_DIM - newsize.width;
				int pad_h = PROCESS_DIM - newsize.height;
				int left = pad_w / 2;
				int right = pad_w - left;
				int top = pad_h / 2;
				int bottom = pad_h - top;
				cv::Mat resized;
				cv::copyMakeBorder(resized_down, resized,
								   top, bottom, left, right,
								   cv::BORDER_CONSTANT,
								   cv::Scalar(114, 114, 114));

				// Prepare the tensor
				tensor3_t *metadata = (tensor3_t*)(worker_buffers[i]);
				metadata->c = 3;
				metadata->w = PROCESS_DIM;
				metadata->h = PROCESS_DIM;
				float *data = (float*)(metadata + 1);

				// Populate the tensor
				// TODO: May need to reverse the channel loop
				for (int c = 0; c < 3; ++c) {
					for (int row = 0; row < PROCESS_DIM; row++) {
						for (int col = 0; col < PROCESS_DIM; col++) {
							// Compute the index in the raw image
							int im_idx = row * PROCESS_DIM * metadata->c +
							col * metadata->c +
							(2 - c);

							// Compute the index in the buffer
							// Organized like this to make splitting easier
							int buf_idx = c * metadata->w * metadata->h +
							row * metadata->w +
							col;

							// Transfer to the buffer. Normalize the colors
							data[buf_idx] = resized.data[im_idx] / 255.0;
						}
					}
				}

				if (i == 0) {
					// To MASTER process, do nothing
					continue;
				} else {
					// To WORKER process, MPI_Send
					MPI_Isend(worker_buffers[i], send_size, MPI_BYTE, i, SEND_FRAME, MPI_COMM_WORLD, &requests[i]);
				}
			}

			// Sends completed
			printf("Process %d: Sent all of frames %d - %d\n", my_rank, frames_parsed, frames_parsed + nframes + 1);

			// Wait for all Isends to resolve
			for (int i = 1; i < nframes; i++) MPI_Wait(&requests[i], MPI_STATUS_IGNORE);

			// Run the model myself with my frame
			out_tensor = detect((tensor3_t*) im_buf, my_rank);
			//printf("Process %d: Test %d\n", out_tensor->c);

			// Decode my own output
			cv::Mat image = frames_buffered[0];
			float r = std::min((float)PROCESS_DIM / (float)image.cols, (float)PROCESS_DIM / (float)image.rows);
			float pad_w = PROCESS_DIM - (r * image.cols);
			float pad_h = PROCESS_DIM - (r * image.rows);

			for (int i = 0; i < out_tensor->c; i++) {
				void *base_ptr = (void*)((char*)(out_tensor + 1) + i * 6);
				int *classid = (int*)(base_ptr);
				float *box_bounds = (float*)((int*)(base_ptr) + 1);
				int cid = classid[0];
				float conf = box_bounds[0];
				int x1 = std::round((box_bounds[1] - (pad_w / 2)) / r);
				int y1 = std::round((box_bounds[2] - (pad_h / 2)) / r);
				int x2 = std::round((box_bounds[3] / r) + x1);
				int y2 = std::round((box_bounds[4] / r) + y1);

				printf("Process %d found Bounding Box from (%d, %d) to (%d, %d) or CID %d with confidence %f\n", my_rank, x1, y1, x2, y2, cid, conf);

				cv::rectangle(
					image,
				  cv::Point(x1, y1- 20),
							  cv::Point(x1 + 100, y1),
							  cv::Scalar(255, 0, 0),
							  -1
				);

				cv::rectangle(
					image,
				  cv::Point(x1, y1),
							  cv::Point(x2, y2),
							  cv::Scalar(255, 0, 0),
							  2
				);

				char out_text[256];
				sprintf(out_text, "id%d, %.2f", cid, conf);

				cv::putText(
					image,
				out_text,
				cv::Point(x1, y1),
							cv::FONT_HERSHEY_SIMPLEX,
				0.5,
				cv::Scalar(255, 255, 255)
				);
			}
			free(out_tensor); // Free as we complete since this is a pointer allocated by the model
			writer.write(image);

			// Decode worker output
			for (int j = 1; j < nframes; j++) {
				// Prepare the frame
				cv::Mat image = frames_buffered[j];
				float r = std::min((float)PROCESS_DIM / (float)image.cols, (float)PROCESS_DIM / (float)image.rows);
				float pad_w = PROCESS_DIM - (r * image.cols);
				float pad_h = PROCESS_DIM - (r * image.rows);

				// Receive the bounding box data
				MPI_Recv(out_buf, output_max_size, MPI_BYTE, j, RECV_FRAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				out_tensor = (tensor3_t*)out_buf;
				printf("Process %d: Received data from worker %d\n", my_rank, j);

				for (int i = 0; i < out_tensor->c; i++) {
					void *base_ptr = (void*)((char*)(out_tensor + 1) + i * 6);
					int *classid = (int*)(base_ptr);
					float *box_bounds = (float*)((int*)(base_ptr) + 1);
					int cid = classid[0];
					float conf = box_bounds[0];
					int x1 = std::round((box_bounds[1] - (pad_w / 2)) / r);
					int y1 = std::round((box_bounds[2] - (pad_h / 2)) / r);
					int x2 = std::round((box_bounds[3] / r) + x1);
					int y2 = std::round((box_bounds[4] / r) + y1);

					printf("Process %d found Bounding Box from (%d, %d) to (%d, %d) or CID %d with confidence %f\n", my_rank, x1, y1, x2, y2, cid, conf);

					cv::rectangle(
						image,
				   cv::Point(x1, y1- 20),
								  cv::Point(x1 + 100, y1),
								  cv::Scalar(255, 0, 0),
								  -1
					);

					cv::rectangle(
						image,
				   cv::Point(x1, y1),
								  cv::Point(x2, y2),
								  cv::Scalar(255, 0, 0),
								  2
					);

					char out_text[256];
					sprintf(out_text, "id%d, %.2f", cid, conf);

					cv::putText(
						image,
				 out_text,
				 cv::Point(x1, y1),
								cv::FONT_HERSHEY_SIMPLEX,
				 0.5,
				 cv::Scalar(255, 255, 255)
					);
				}

				//cv::imwrite("output.jpg", image);
				writer.write(image);
			}

			// Finished Writing all frames
			printf("Process %d: Wrote all of frames %d - %d\n", my_rank, frames_parsed, frames_parsed + nframes + 1);
			frames_parsed += nframes;
		}

		// Release all resources
		cap.release();
		writer.release();
		free(out_buf);
		free(send_buffer);
	} else {
		size_t recv_size = sizeof(tensor3_t) + sizeof(float) * PROCESS_DIM * PROCESS_DIM * 3;
		size_t output_max_size = sizeof(tensor3_t) + sizeof(int) * 8400 * 6;
		MPI_Request req;
		tensor3_t *out_tensor;
		while (true) {
			// Receive
			MPI_Irecv((void*)im_buf, recv_size, MPI_BYTE, 0, SEND_FRAME, MPI_COMM_WORLD, &req);
			MPI_Wait(&req, MPI_STATUS_IGNORE);

			// Check terminator
			if (((tensor3_t*)im_buf)->c == -1) {
				printf("Process %d: Acknowledged, no more frames to receive\n", my_rank);
				break;
			}

			// Process the buffer
			out_tensor = detect((tensor3_t*) im_buf, my_rank);

			// Send the buffer back
			MPI_Send(out_tensor, output_max_size, MPI_BYTE, 0, RECV_FRAME, MPI_COMM_WORLD);
			free(out_tensor);
		}
	}

	/*
	 / /* Load the Image
	 load_image(filepath, 0, im_buf, my_rank);

	 cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
	 float r = std::min((float)PROCESS_DIM / (float)image.cols, (float)PROCESS_DIM / (float)image.rows);
	 float pad_w = PROCESS_DIM - (r * image.cols);
	 float pad_h = PROCESS_DIM - (r * image.rows);

	 // Run YOLOv8
	 out_tensor = detect((tensor3_t*) im_buf, my_rank);

	 // Break down the output
	 for (int i = 0; i < out_tensor->c; i++) {
		 void *base_ptr = (void*)((char*)(out_tensor + 1) + i * 6);
		 int *classid = (int*)(base_ptr);
		 float *box_bounds = (float*)((int*)(base_ptr) + 1);
		 int cid = classid[0];
		 float conf = box_bounds[0];
		 int x1 = std::round((box_bounds[1] - (pad_w / 2)) / r);
		 int y1 = std::round((box_bounds[2] - (pad_h / 2)) / r);
		 int x2 = std::round((box_bounds[3] / r) + x1);
		 int y2 = std::round((box_bounds[4] / r) + y1);

		 printf("Process %d found Bounding Box from (%d, %d) to (%d, %d) or CID %d with confidence %f\n", my_rank, x1, y1, x2, y2, cid, conf);

		 cv::rectangle(
			 image,
			 cv::Point(x1, y1- 20),
			 cv::Point(x1 + 100, y1),
			 cv::Scalar(255, 0, 0),
			 -1
			 );

			 cv::rectangle(
				 image,
				 cv::Point(x1, y1),
				 cv::Point(x2, y2),
				 cv::Scalar(255, 0, 0),
				 2
				 );

				 char out_text[256];
				 sprintf(out_text, "id%d, %.2f", cid, conf);

				 cv::putText(
					 image,
					 out_text,
					 cv::Point(x1, y1),
					 cv::FONT_HERSHEY_SIMPLEX,
					 0.5,
					 cv::Scalar(255, 255, 255)
					 );
					 cv::imwrite("output.jpg", image);
}

*/

	// Cleanup
	free(im_buf);
	free_model(my_rank);
	free(conv_buf);

	MPI_Finalize();
	return 0;
}

size_t compute_conv_alloc_amt() {
	// Allocate memory for all the convolution layers
	size_t conv_block_size = 0;
	conv_block_size += conv_malloc_amt(3, 3, 16); 		// Conv2D 0
	conv_block_size += conv_malloc_amt(3, 16, 32);		// Conv2D 1
	conv_block_size += c2f_malloc_amt(32, 32, 1);		// C2f 2
	conv_block_size += conv_malloc_amt(3, 32, 64);		// Conv2D 3
	conv_block_size += c2f_malloc_amt(64, 64, 2);		// C2f 4
	conv_block_size += conv_malloc_amt(3, 64, 128);		// Conv2D 5
	conv_block_size += c2f_malloc_amt(128, 128, 2);		// C2f 6
	conv_block_size += conv_malloc_amt(3, 128, 256);	// Conv2D 7
	conv_block_size += c2f_malloc_amt(256, 256, 1);		// C2f 8
	conv_block_size += sppf_malloc_amt(256);		// SPPF 9
	conv_block_size += c2f_malloc_amt(384, 128, 1);		// C2f 12
	conv_block_size += c2f_malloc_amt(192, 64, 1);		// C2f 15
	conv_block_size += conv_malloc_amt(3, 64, 64);		// Conv 16
	conv_block_size += c2f_malloc_amt(192, 128, 1);		// C2f 18
	conv_block_size += conv_malloc_amt(3, 128, 128);	// Conv 19
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21

	// TODO FIX THIS
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21

	return conv_block_size;
}
