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

