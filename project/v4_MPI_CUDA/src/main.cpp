#include <iostream>

#include <mpi.h>

#include "layer_structs.h"
#include "model.h"

#define		ROOT		0

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

	if (my_rank == ROOT) {}

	// Cleanup
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
