# CS 455 GPU Cluster Programming Project
*This project was developed by Sebastian Vargas and Aaron Agcaoili for NJIT's CS455 GPU Cluster Programming course.*

---

## Project: YOLOv8 Implementation with MPI + CUDA
The goal of this project is to construct a rough implementation of the YOLOv8 image detector model using MPI and CUDA (hopefully, CUDA-aware MPI towards the end for increased efficiency), and to compare our implementation with the existing implementation of the model both in efficiency of training passes and validation passes.

This `README.md` file will contain a chronological detailing/documentation of the implementation of this architecture.

### Project Dependencies, Compilation, and Execution
This project was programmed and executed on two devices running the `Fedora 41` operating system. For this project we used version **12.9** of the CUDA Toolkit as that was the version compatible with the compute capability of both of our devices.

In addition, this project relies on the following libraries (paired with their versions):
```
openmpi-5.0.5-2.fc41
openmpi-devel-5.0.5-2.fc41
opencv-4.10.0-4.fc41
opencv-devel-4.10.0-4.fc41
```

To compile our programs we opted to use the **CMake** meta build system. Hence this project includes the `CMakeLists.txt` files needed in order to compile the executables for this project. To compile the project simply run the following commands:
```
mkdir build
cmake -S . -B build
cmake --build build
```
The first two lines in the above sequence need only be run the first time the project is built.

The build folder is organized into the same folder hierarchy as the source code for this project, such that each version of executable for this project can be found under its corresponding subfolder.

Each of the implementations are contained in the following folders and take the following parameters
```
./build/project/v1_serial/YOLOv8_CPP_Serial_Main <CONFIDENCE_THRESHOLD> <IOU_THRESHOLD> <FILEPATH>
./build/project/v2_MPI_only/YOLOv8_CPP_MPI <CONFIDENCE_THRESHOLD> <IOU_THRESHOLD> <FILEPATH>
./build/project/v4_MPI_CUDA/YOLOv8_CPP_CUDA_MPI <FILEPATH> <CONFIDENCE_THRESHOLD> <IOU THRESHOLD>
```
We recommand `0.2` in place of `<CONFIDENCE_THRESHOLD>` and `0.3` in place of <IOU_THREHSHOLD>`

## Project Timeline
### v0 - Python Implementation, Model Exploration, and Tooling
The `v0` version of this project is not so much an implementation of the architecture as it is an exploration of the architecture. We took some time to both download a version of the model and analyze the architecture to understand how model parameters and hyperparameters were chosen, and determine the best approach for implementing the model in C++ as opposed to Python.

Ultralytics provides pretrained versions of their detection models, so we use a pretrained version of the YOLOv8 nano model (containing around 3.1 million parameters) which was pretrained on the COCOv8 dataset. The `ipynb` file contained demonstrates how the model works and what output we expect to come out of the model.

We realized given time contraints for this project and hardware constraints of our devices that it would not be feasible in this iteration of the project to implement a backward pass for our model, so we focused on implementing and optimizing the forward pass for our model, and our plan is to use the Python implementation as a baseline of comparison/performance testing for our code.

In order for our implementation of the architecture to perform well, however, we would need to be able to extract the weights of a pretrained model and format them into a parseable format for our implementation of the YOLOv8 model. Thus the `ipynb` file in this iteration of the project includes a set of routines to extract model weights and biases from a pretrained YOLOv8 model, and write them to a raw binary file which will be parsed as part of the preprocessing pipeline for our version of the model.

### v1 - Serial Implementation
One of the most challenging components of the serial implementation was the overall pipeline implementation.

For the serial convolution implementation, we opted for a rough and simple implementation of the convolution operation (with more time we would ideally reformat the operation as a matrix multiply). Since the bulk of the challenge for drafting the model in the serial version is setting up pipelines for routing and pushing data to memory, we focused on getting a model that functioned with a pipeline that would be feasible to modify to include MPI and CUDA functionality later.




