#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <mpi.h>

#include "prepostproc.h"
#include "layer_structs.h"
#include "serial_funcs.h"

#define 	MAX_TENSOR_BLOCK	(640 * 640 * 32 + 8)
#define		PREALLOC_TENSORS	20

#define		MB_UNIT			(2 << 20)

#define		MAX_LAYERS		50

#define		TOTAL_BOXES			8400

void process_tile(int);

void test_tensor_printf(int rank, tensor3_t* tensor, int x, int y, int channel) {
	printf("Process %d: Tensor is %d x %d with %d channels\n", rank,
			tensor->w,
			tensor->h,
			tensor->c);

	
	printf("\tValue at (%d, %d, %d) is %f\n",
			channel,
			x,
			y,
			tensor->data[
				channel * tensor->w * tensor->h + 
				y * tensor->w + 
				x
			]
		);
}

// Global variables for the image and convolutional buffers
void *conv_buf, *img_buf;
tensor3_t *blocks[PREALLOC_TENSORS];
conv_t *conv_kernels[64];

int main(int argc, char *argv[]) {
	// MPI Initialization
	int my_rank, nprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	printf("Initialized process %d out of %d processes\n", my_rank, nprocs);

	int tot_tiles = 1, tile_dim = 1;
	while (tot_tiles < nprocs) {
		tot_tiles <<= 2;
		tile_dim <<= 1;
	}

	int *work_per_proc = (int*)malloc(sizeof(int) * nprocs); // For gatherv
	for (int i = 0; i < nprocs; i++) {
		work_per_proc[i] = tot_tiles / nprocs;
		if (i < (tot_tiles % nprocs)) work_per_proc[i]++;
	}

	// Read from argv
	float THRESHOLD = atof(argv[1]);
	float IOU_THRESHOLD = atof(argv[2]);
	char fpath_buf[512];
	strcpy(fpath_buf, argv[3]);

	// Allocate a big buffer
	img_buf = calloc(MAX_TENSOR_BLOCK * PREALLOC_TENSORS, sizeof(float));
	
	// Partition the big buffer into tensor blocks we can reuse over and over
	for (int i = 0; i < PREALLOC_TENSORS; i++)
		blocks[i] = (tensor3_t*)((float*)img_buf + (MAX_TENSOR_BLOCK * i));
	printf("Process %d Allocated %d max size tensors, worth %d MiB\n", my_rank,
			PREALLOC_TENSORS,
			MAX_TENSOR_BLOCK * sizeof(float) * PREALLOC_TENSORS / MB_UNIT);

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

	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	conv_block_size += c2f_malloc_amt(384, 256, 1);		// C2f 21
	
	printf("Process %d Allocating %d MiB of memory for conv layers\n", my_rank, conv_block_size / MB_UNIT);
	conv_buf = calloc(conv_block_size / sizeof(float), sizeof(float));

	// TODO: Read convolution layers and send them to the other processes as we are reading
	FILE *infile = fopen("./filters.bin", "rb");
	void *curr = conv_buf;
	int curr_kernel = 0;
	while (1) {
		// Exit out if we hit the EOF
		if (curr == NULL) break;

		// Save the kernel pointer
		conv_kernels[curr_kernel] = (conv_t*)curr;
		curr = fread_conv(infile, curr);

		curr_kernel++;
	}
	fclose(infile);
	printf("Process %d has read all convolutional layers from the binary file\n", my_rank);

	// Load the image
	tensor3_t *raw_tiles;
	int ntiles;
	load_image_work(fpath_buf, 1, my_rank, nprocs, &raw_tiles, &ntiles);
	int len_per_tensor = sizeof(tensor3_t) + raw_tiles->w * raw_tiles->h * raw_tiles->c * sizeof(float);

	// Create the boxes buffer
	bbox_t *boxes = NULL;
	int box_counter = 0, box_offset = 0;

	for (int i = 0; i < ntiles; i++) {
		// Ensure that all tensors we are working with are received
		tensor3_t* curr_tensor = (tensor3_t*)((char*)raw_tiles + i * len_per_tensor);
		test_tensor_printf(my_rank, curr_tensor, 1, 1, 0);

		// Copy the appropriate tensor
		memcpy((void*)blocks[0], (void*)curr_tensor, len_per_tensor);

		// Save the dimensionality of the image (w/o padding)
		int base_dim = blocks[0]->w - 2;

		// Run
		process_tile(my_rank);
		printf("Process %d finished processing a tile\n", my_rank);

		// Extract the boxes
		box_counter += blocks[7]->w * blocks[7]->h +
		blocks[8]->w * blocks[8]->h +
		blocks[9]->w * blocks[9]->h;
		printf("Process %d produced %d boxes as of this iteration\n", my_rank, box_counter);
		if (!boxes) {
			boxes = (bbox_t*)calloc(box_counter * tot_tiles, sizeof(bbox_t)); // We'll use gatherv later
		}

		// Compute the x and y tile offset for boxes
		int tile_id = 0;
		for (int j = 0; j < my_rank; j++) tile_id += work_per_proc[j];
		tile_id += i;
		int tile_x = tile_id % tile_dim;
		int tile_y = tile_id / tile_dim;
		int slide_x = (PROCESS_DIM - base_dim) / (tile_dim - 1);
		int slide_y = (PROCESS_DIM - base_dim) / (tile_dim - 1);
		int tile_x_offset = tile_x * slide_x;
		int tile_y_offset = tile_y * slide_y;

		float bins[16];
		float classes[80];
		float softmax_sum, expected;
		int box_center_x, box_center_y, stride;
		stride = base_dim / blocks[7]->h / 2; // Just use h for now honestly, assume square so no matter

		// Iterate
		for (int y = 0; y < blocks[7]->h; y++)
			for (int x = 0; x < blocks[7]->w; x++) {

				softmax_sum = 0.0;
				expected = 0.0;

				box_center_x = x * stride * 2 + stride;
				box_center_y = y * stride * 2 + stride;

				// Start with the left pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[7]->data[
						c * blocks[7]->w * blocks[7]->h +
						y * blocks[7]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}

				boxes[box_offset + y * blocks[7]->w + x].x = box_center_x - int(round(expected * stride * 2));

				softmax_sum = 0.0;
				expected = 0.0;
				// Then the top pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[7]->data[
						(c + 16) * blocks[7]->w * blocks[7]->h +
						y * blocks[7]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}
				boxes[box_offset + y * blocks[7]->w + x].y = box_center_y - int(round(expected * stride * 2));

				softmax_sum = 0.0;
				expected = 0.0;
				// Then the right pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[7]->data[
						(c + 32) * blocks[7]->w * blocks[7]->h +
						y * blocks[7]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}
				boxes[box_offset + y * blocks[7]->w + x].w = box_center_x + int(round(expected * stride * 2)) - boxes[box_offset + y * blocks[7]->w + x].x;

				softmax_sum = 0.0;
				expected = 0.0;
				// Then the top pixel
				for (int c = 0; c < 16; ++c) {
					bins[c] = blocks[7]->data[
						(c + 48) * blocks[7]->w * blocks[7]->h +
						y * blocks[7]->w +
						x
					];

					softmax_sum += exp(bins[c]);
				}
				for (int c = 0; c < 16; ++c) {
					bins[c] = exp(bins[c]) / softmax_sum;
					expected += bins[c] * c;
				}
				boxes[box_offset + y * blocks[7]->w + x].h = box_center_y + int(round(expected * stride * 2)) - boxes[box_offset + y * blocks[7]->w + x].y;


				// Class Selection
				softmax_sum = 0.0;
				expected = 0.0;
				for (int c = 0; c < 80; ++c) {
					classes[c] = blocks[10]->data[
						c * blocks[10]->w * blocks[10]->h +
						y * blocks[10]->w +
						x
					];

					softmax_sum += exp(classes[c]);
				}
				for (int c = 0; c < 80; ++c) {
					classes[c] = 1.0f / (1.0f + expf(-classes[c]));
					if (classes[c] > expected) {
						expected = classes[c];
						boxes[box_offset + y * blocks[7]->w + x].cid = c;
						boxes[box_offset + y * blocks[7]->w + x].class_conf = classes[c];
					}
				}
				boxes[box_offset + y * blocks[7]->w + x].x += tile_x_offset;
				boxes[box_offset + y * blocks[7]->w + x].y += tile_y_offset;
			}

			int offset = blocks[7]->w * blocks[7]->h;
			stride = base_dim / blocks[8]->h / 2; // Just use h for now honestly, assume square so no matter

			// Iterate
			for (int y = 0; y < blocks[8]->h; y++)
				for (int x = 0; x < blocks[8]->w; x++) {

					softmax_sum = 0.0;
					expected = 0.0;

					box_center_x = x * stride * 2 + stride;
					box_center_y = y * stride * 2 + stride;

					// Start with the left pixel
					for (int c = 0; c < 16; ++c) {
						bins[c] = blocks[8]->data[
							c * blocks[8]->w * blocks[8]->h +
							y * blocks[8]->w +
							x
						];

						softmax_sum += exp(bins[c]);
					}
					for (int c = 0; c < 16; ++c) {
						bins[c] = exp(bins[c]) / softmax_sum;
						expected += bins[c] * c;
					}
					boxes[box_offset + offset + y * blocks[8]->w + x].x = box_center_x - int(round(expected * stride * 2));

					softmax_sum = 0.0;
					expected = 0.0;
					// Then the top pixel
					for (int c = 0; c < 16; ++c) {
						bins[c] = blocks[8]->data[
							(c + 16) * blocks[8]->w * blocks[8]->h +
							y * blocks[8]->w +
							x
						];

						softmax_sum += exp(bins[c]);
					}
					for (int c = 0; c < 16; ++c) {
						bins[c] = exp(bins[c]) / softmax_sum;
						expected += bins[c] * c;
					}
					boxes[box_offset + offset + y * blocks[8]->w + x].y = box_center_y - int(round(expected * stride * 2));

					softmax_sum = 0.0;
					expected = 0.0;
					// Then the right pixel
					for (int c = 0; c < 16; ++c) {
						bins[c] = blocks[8]->data[
							(c + 32) * blocks[8]->w * blocks[8]->h +
							y * blocks[8]->w +
							x
						];

						softmax_sum += exp(bins[c]);
					}
					for (int c = 0; c < 16; ++c) {
						bins[c] = exp(bins[c]) / softmax_sum;
						expected += bins[c] * c;
					}
					boxes[box_offset + offset + y * blocks[8]->w + x].w = box_center_x + int(round(expected * stride * 2)) - boxes[box_offset + offset + y * blocks[8]->w + x].x;

					softmax_sum = 0.0;
					expected = 0.0;
					// Then the top pixel
					for (int c = 0; c < 16; ++c) {
						bins[c] = blocks[8]->data[
							(c + 48) * blocks[8]->w * blocks[8]->h +
							y * blocks[8]->w +
							x
						];

						softmax_sum += exp(bins[c]);
					}
					for (int c = 0; c < 16; ++c) {
						bins[c] = exp(bins[c]) / softmax_sum;
						expected += bins[c] * c;
					}
					boxes[box_offset + offset + y * blocks[8]->w + x].h = box_center_y + int(round(expected * stride * 2)) - boxes[box_offset + offset + y * blocks[8]->w + x].y;


					// Class Selection
					softmax_sum = 0.0;
					expected = 0.0;
					for (int c = 0; c < 80; ++c) {
						classes[c] = blocks[11]->data[
							c * blocks[11]->w * blocks[11]->h +
							y * blocks[11]->w +
							x
						];

						softmax_sum += exp(classes[c]);
					}
					for (int c = 0; c < 80; ++c) {
						classes[c] = 1.0f / (1.0f + expf(-classes[c]));
						if (classes[c] > expected) {
							expected = classes[c];
							boxes[box_offset + offset + y * blocks[8]->w + x].cid = c;
							boxes[box_offset + offset + y * blocks[8]->w + x].class_conf = classes[c];
						}
					}
					boxes[box_offset + offset + y * blocks[8]->w + x].x += tile_x_offset;
					boxes[box_offset + offset + y * blocks[8]->w + x].y += tile_y_offset;
				}

				offset += blocks[8]->w * blocks[8]->h;
				stride = base_dim / blocks[9]->h / 2; // Just use h for now honestly, assume square so no matter

				// Iterate
				for (int y = 0; y < blocks[9]->h; y++)
					for (int x = 0; x < blocks[9]->w; x++) {

						softmax_sum = 0.0;
						expected = 0.0;

						box_center_x = x * stride * 2 + stride;
						box_center_y = y * stride * 2 + stride;

						// Start with the left pixel
						for (int c = 0; c < 16; ++c) {
							bins[c] = blocks[9]->data[
								c * blocks[9]->w * blocks[9]->h +
								y * blocks[9]->w +
								x
							];

							softmax_sum += exp(bins[c]);
						}
						for (int c = 0; c < 16; ++c) {
							bins[c] = exp(bins[c]) / softmax_sum;
							expected += bins[c] * c;
						}
						boxes[box_offset + offset + y * blocks[9]->w + x].x = box_center_x - int(round(expected * stride * 2));

						softmax_sum = 0.0;
						expected = 0.0;
						// Then the top pixel
						for (int c = 0; c < 16; ++c) {
							bins[c] = blocks[9]->data[
								(c + 16) * blocks[9]->w * blocks[9]->h +
								y * blocks[9]->w +
								x
							];

							softmax_sum += exp(bins[c]);
						}
						for (int c = 0; c < 16; ++c) {
							bins[c] = exp(bins[c]) / softmax_sum;
							expected += bins[c] * c;
						}
						boxes[box_offset + offset + y * blocks[9]->w + x].y = box_center_y - int(round(expected * stride * 2));

						softmax_sum = 0.0;
						expected = 0.0;
						// Then the right pixel
						for (int c = 0; c < 16; ++c) {
							bins[c] = blocks[9]->data[
								(c + 32) * blocks[9]->w * blocks[9]->h +
								y * blocks[9]->w +
								x
							];

							softmax_sum += exp(bins[c]);
						}
						for (int c = 0; c < 16; ++c) {
							bins[c] = exp(bins[c]) / softmax_sum;
							expected += bins[c] * c;
						}
						boxes[box_offset + offset + y * blocks[9]->w + x].w = box_center_x + int(round(expected * stride * 2)) - boxes[box_offset + offset + y * blocks[9]->w + x].x;

						softmax_sum = 0.0;
						expected = 0.0;
						// Then the top pixel
						for (int c = 0; c < 16; ++c) {
							bins[c] = blocks[9]->data[
								(c + 48) * blocks[9]->w * blocks[9]->h +
								y * blocks[9]->w +
								x
							];

							softmax_sum += exp(bins[c]);
						}
						for (int c = 0; c < 16; ++c) {
							bins[c] = exp(bins[c]) / softmax_sum;
							expected += bins[c] * c;
						}
						boxes[box_offset + offset + y * blocks[9]->w + x].h = box_center_y + int(round(expected * stride * 2)) - boxes[box_offset + offset + y * blocks[9]->w + x].y;

						// Class Selection
						softmax_sum = 0.0;
						expected = 0.0;
						for (int c = 0; c < 80; ++c) {
							classes[c] = blocks[12]->data[
								c * blocks[12]->w * blocks[12]->h +
								y * blocks[12]->w +
								x
							];

							softmax_sum += exp(classes[c]);
						}
						for (int c = 0; c < 80; ++c) {
							classes[c] = 1.0f / (1.0f + expf(-classes[c]));
							if (classes[c] > expected) {
								expected = classes[c];
								boxes[box_offset + offset + y * blocks[9]->w + x].cid = c;
								boxes[box_offset + offset + y * blocks[9]->w + x].class_conf = classes[c];
							}
						}
						boxes[box_offset + offset + y * blocks[9]->w + x].x += tile_x_offset;
						boxes[box_offset + offset + y * blocks[9]->w + x].y += tile_y_offset;
					}

					// Update the offset once we've generated all the boxes for this run
					box_offset = box_counter;
	}

	// Merge results
	printf("Process %d finished creating boxes\n", my_rank);
	if (my_rank == 0) {
		int next_recv_amt = 0;
		for (int i = 1; i < nprocs; i++) {
			// Receive the number of items we are expecting
			MPI_Recv(&next_recv_amt, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("MASTER is expecting %d from process %d\n", next_recv_amt, i);
			// Pull the boxes
			MPI_Recv((void*)(boxes + box_counter), next_recv_amt * sizeof(bbox_t), MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("MASTER received %d from process %d\n", next_recv_amt, i);

			// Update the box box_counter
			box_counter += next_recv_amt;
		}
		printf("MASTER has %d boxes in total\n", box_counter);
	} else {
		// Communicate how many boxes I have
		MPI_Send(&box_counter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		printf("%d is sending %d boxes\n", my_rank, box_counter);

		// Send the boxes
		MPI_Send((void*)boxes, box_counter * sizeof(bbox_t), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
		printf("%d sent %d boxes\n", my_rank, box_counter);
	}

	if (my_rank == 0) {
		// Initialize inactive flags and build list of candidate indices
		std::vector<int> idxs;
		idxs.reserve(box_counter);
		for (int i = 0; i < box_counter; ++i) {
			if (boxes[i].class_conf >= THRESHOLD) {

				boxes[i].inactive = 0;
				idxs.push_back(i);
			} else {
				//printf("%f\n", boxes[i].class_conf);
				boxes[i].inactive = 1;
			}
		}

		// Sort by confidence high -> low
		std::sort(idxs.begin(), idxs.end(),
				  [&](int a, int b) {
					  return boxes[a].class_conf > boxes[b].class_conf;
				  });

		// Standard NMS: for each box, suppress later boxes with high IoU and same class
		for (size_t m = 0; m < idxs.size(); ++m) {
			int i = idxs[m];
			if (boxes[i].inactive) continue;

			for (size_t n = m + 1; n < idxs.size(); ++n) {
				int j = idxs[n];
				if (boxes[j].inactive) continue;

				// Optional: per-class NMS
				if (boxes[i].cid != boxes[j].cid) continue;

				float x_int_min = std::max(boxes[i].x, boxes[j].x);
				float y_int_min = std::max(boxes[i].y, boxes[j].y);
				float x_int_max = std::min(boxes[i].x + boxes[i].w, boxes[j].x + boxes[j].w);
				float y_int_max = std::min(boxes[i].y + boxes[i].h, boxes[j].y + boxes[j].h);
				float inter_area = std::max(0.0f, x_int_max - x_int_min) * std::max(0.0f, y_int_max - y_int_min);

				float area1 = boxes[i].w * boxes[i].h;
				float area2 = boxes[j].w * boxes[j].h;
				float union_area = area1 + area2 - inter_area;

				float iou = inter_area / union_area;
				if (iou > IOU_THRESHOLD) {
					boxes[j].inactive = 1;
				}
			}
		}

		cv::Mat image = cv::imread(fpath_buf, cv::IMREAD_COLOR);
		//cv::Size newsize(PROCESS_DIM, PROCESS_DIM);
		//cv::Mat resized;
		//cv::resize(image, resized, newsize);
		float r = std::min((float)PROCESS_DIM / (float)image.cols, (float)PROCESS_DIM / (float)image.rows);
		float pad_w = PROCESS_DIM - (r * image.cols);
		float pad_h = PROCESS_DIM - (r * image.rows);

		// Debug print: how many were suppressed
		int disabled = 0;
		for (int i = 0; i < box_counter; ++i) {
			if (boxes[i].inactive) disabled++;

			if (!boxes[i].inactive) {
				int x1 = std::round((boxes[i].x - (pad_w / 2)) / r);
				int y1 = std::round((boxes[i].y - (pad_h / 2)) / r);
				int x2 = std::round((boxes[i].w / r) + x1);
				int y2 = std::round((boxes[i].h / r) + y1);

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
				sprintf(out_text, "id%d, %.2f", boxes[i].cid, boxes[i].class_conf);

				cv::putText(
					image,
				out_text,
				cv::Point(x1, y1),
							cv::FONT_HERSHEY_SIMPLEX,
				0.5,
				cv::Scalar(255, 255, 255)
				);
			}


		}
		cv::imwrite("output.jpg", image);
		printf("Process %d Disabled %d boxes\n", my_rank, disabled);
	}


	// Cleanup
	free(boxes);
	free(work_per_proc);
	free(raw_tiles);
	free(conv_buf);
	free(img_buf);

	MPI_Finalize();
}

void process_tile(int rank) {
	// Test the first layer
	conv_t *conv_lay = (conv_t*)conv_buf;
	//printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[0], (tensor3_t*)blocks[1], conv_lay, 1, 0);
	//printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[1], 1, 1, 0);

	// Test the second layer
	//printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[1], (tensor3_t*)blocks[2], conv_kernels[1], 0, 0);
	//printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[2], 0, 0, 0);

	// Test the first c2f
	//printf("\nComputing a C2f\n");
	c2f_layer_serial(
			(tensor3_t**)(blocks + 2),
			&conv_kernels[2],
			1,
			1,
			1);
	//printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[2], 1, 1, 0);

	// Conv
	//printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[2], (tensor3_t*)blocks[0], conv_kernels[6], 0, 0);
	//printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[0], 0, 0, 0);

	// Clean out
	memset((tensor3_t*)blocks[1], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Test the first c2f
	//printf("\nComputing a C2f\n");
	tensor3_t* large_bb_out = c2f_layer_serial(
			(tensor3_t**)(blocks),
			&conv_kernels[7],
			2,
			1,
			1);
	//printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[0], 1, 1, 0);

	// Clean out
	memset((tensor3_t*)blocks[1], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Conv
	//printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[0], (tensor3_t*)blocks[1], conv_kernels[13], 0, 0);
	//printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[1], 0, 0, 0);

	// Test the first c2f
	//printf("\nComputing a C2f\n");
	tensor3_t* med_bb_out = c2f_layer_serial(
			(tensor3_t**)(blocks + 1),
			&conv_kernels[14],
			2,
			1,
			1);
	//printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[1], 1, 1, 0);

	// Clean out
	memset((tensor3_t*)blocks[2], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Conv
	//printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[1], (tensor3_t*)blocks[2], conv_kernels[20], 0, 0);
	//printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[2], 0, 0, 0);

	// Test the first c2f
	//printf("\nComputing a C2f\n");
	c2f_layer_serial(
			(tensor3_t**)(blocks + 2),
			&conv_kernels[21],
			1,
			0,
			1);
	//printf("Completed! Here's some data about the output of the c2f:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[2], 0, 0, 0);

	// Clean out
	memset((tensor3_t*)blocks[3], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Test the SPPF
	//printf("\nPerforming SPPF\n");
	tensor3_t* small_bb_out = sppf_layer_serial(
			(tensor3_t**)(blocks + 2),
			&conv_kernels[25],
			0
	);
	//printf("Completed! Here's some data about the output of sppf:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[2], 0, 0, 0);

	// Clean out
	memset((tensor3_t*)blocks[3], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Upsample + Concat
	int base_channels;
	tensor3_t* concat_out;

	//printf("\nUpsampling and Concatenating\n");
	concat_out = (tensor3_t*)blocks[3];
	upsample_layer_serial(small_bb_out, concat_out);
	base_channels = concat_out->c;
	concat_out->c += med_bb_out->c;

	for (int c = 0; c < med_bb_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * med_bb_out->w * med_bb_out->h +
			(row + 1) * med_bb_out->w + // This one has padding specifically
			(col + 1); // This one has padding specifically

		concat_out->data[out_idx] = med_bb_out->data[in_idx];
	}
	test_tensor_printf(rank, concat_out, 0, 0, 256);

	//printf("\nC2f for the first medium sized head output\n");
	tensor3_t* medium_h1_out = c2f_layer_serial(&blocks[3], &conv_kernels[27], 1, 0, 0);
	test_tensor_printf(rank, concat_out, 1, 1, 0);

	// Clean out
	memset(blocks[4], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Upsample + Concat
	//printf("\nUpsampling and Concatenating\n");
	concat_out = (tensor3_t*)blocks[4];
	upsample_layer_serial(medium_h1_out, concat_out);
	base_channels = concat_out->c;
	concat_out->c += large_bb_out->c;

	for (int c = 0; c < large_bb_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * large_bb_out->w * large_bb_out->h +
			(row + 1) * large_bb_out->w + // This one has padding specifically
			(col + 1); // This one has padding specifically

		concat_out->data[out_idx] = large_bb_out->data[in_idx];
	}
	test_tensor_printf(rank, concat_out, 0, 0, 0);

	//printf("\nC2f for the first large sized head output\n");
	tensor3_t* large_h1_out = c2f_layer_serial(&blocks[4], &conv_kernels[31], 1, 1, 0);
	test_tensor_printf(rank, large_h1_out, 1, 1, 0);

	// Conv layer
	//printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[4], (tensor3_t*)blocks[5], conv_kernels[35], 0, 0);
	//printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[5], 0, 0, 0);

	// Concat into C2f
	concat_out = (tensor3_t*)blocks[5];
	base_channels = concat_out->c;
	concat_out->c += medium_h1_out->c;

	for (int c = 0; c < medium_h1_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * medium_h1_out->w * medium_h1_out->h +
			row * medium_h1_out->w + // This one has padding specifically
			col; // This one has padding specifically

		concat_out->data[out_idx] = medium_h1_out->data[in_idx];
	}

	//printf("\nC2f for the medium sized head output\n");
	tensor3_t* medium_head_out = c2f_layer_serial(&blocks[5], &conv_kernels[36], 1, 1, 0);
	test_tensor_printf(rank, concat_out, 1, 1, 0);

	// Clean out
	memset(blocks[6], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	// Conv layer
	//printf("\nComputing a convolution\n");
	conv_layer_serial((tensor3_t*)blocks[5], (tensor3_t*)blocks[6], conv_kernels[40], 0, 0);
	//printf("Completed! Here's some data about the output of the conv:\n");
	test_tensor_printf(rank, (tensor3_t*)blocks[6], 0, 0, 0);

	// Concat into C2f
	concat_out = (tensor3_t*)blocks[6];
	base_channels = concat_out->c;
	concat_out->c += small_bb_out->c;

	for (int c = 0; c < small_bb_out->c; ++c)
	for (int row = 0; row < concat_out->h; row++)
	for (int col = 0; col < concat_out->w; col++) {
		// Compute output index
		int out_idx = (c + base_channels) * concat_out->w * concat_out->h +
			row * concat_out->w +
			col;

		int in_idx = c * small_bb_out->w * small_bb_out->h +
			row * small_bb_out->w + // This one has padding specifically
			col; // This one has padding specifically

		concat_out->data[out_idx] = small_bb_out->data[in_idx];
	}

	//printf("\nC2f for the small sized head output\n");
	tensor3_t* small_head_out = c2f_layer_serial(&blocks[6], &conv_kernels[41], 1, 1, 0);
	test_tensor_printf(rank, concat_out, 1, 1, 0);

	// Clean out
	memset(blocks[7], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));

	/*
	printf("\nOccupied Addresses:\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n%p = %p\n",
			large_bb_out, blocks[0],
			med_bb_out, blocks[1],
			small_bb_out, blocks[2],
			medium_h1_out, blocks[3],
			large_h1_out, blocks[4],
			medium_head_out, blocks[5],
			small_head_out, blocks[6]);
	*/

	// Detect 1
	//printf("\nBBox Detect 64 Channel\n");
	//printf("Input: %d %d %d\n", large_h1_out->w, large_h1_out->h, large_h1_out->c);
	detect_layer_serial(large_h1_out, &blocks[7], &conv_kernels[45]);
	memset(blocks[8], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(rank, blocks[7], 0, 0, 0);

	// Detect 2
	//printf("\nBBox Detect 128 Channel\n");
	//printf("Input: %d %d %d\n", medium_head_out->w, medium_head_out->h, medium_head_out->c);
	detect_layer_serial(medium_head_out, &blocks[8], &conv_kernels[48]);
	memset(blocks[9], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(rank, blocks[8], 0, 0, 0);

	// Detect 3
	//printf("\nBBox Detect 256 Channel\n");
	//printf("Input: %d %d %d\n", small_head_out->w, small_head_out->h, small_head_out->c);
	detect_layer_serial(small_head_out, &blocks[9], &conv_kernels[51]);
	memset(blocks[10], 0, 8 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(rank, blocks[9], 0, 0, 0);

	// Detect 1
	//printf("\nClass Detect 64 Channel\n");
	//printf("Input: %d %d %d\n", large_h1_out->w, large_h1_out->h, large_h1_out->c);
	detect_layer_serial(large_h1_out, &blocks[10], &conv_kernels[54]);
	memset(blocks[11], 0, 4 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(rank, blocks[10], 0, 0, 0);

	// Detect 2
	//printf("\nClass Detect 128 Channel\n");
	//printf("Input: %d %d %d\n", medium_head_out->w, medium_head_out->h, medium_head_out->c);
	detect_layer_serial(medium_head_out, &blocks[11], &conv_kernels[57]);
	memset(blocks[12], 0, 4 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(rank, blocks[11], 0, 0, 0);
	//test_tensor_printf(blocks[11], 1, 0, 0);
	//test_tensor_printf(blocks[11], 2, 0, 0);

	// Detect 3
	//printf("\nBBox Detect 256 Channel\n");
	//printf("Input: %d %d %d\n", small_head_out->w, small_head_out->h, small_head_out->c);
	detect_layer_serial(small_head_out, &blocks[12], &conv_kernels[60]);
	memset(blocks[13], 0, 4 * MAX_TENSOR_BLOCK * sizeof(float));
	test_tensor_printf(rank, blocks[12], 0, 0, 0);
}

