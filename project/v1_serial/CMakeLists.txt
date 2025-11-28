cmake_minimum_required(VERSION 3.15)
project(YOLOv8_CPP VERSION 0.1
	DESCRIPTION ""
	LANGUAGES CXX CUDA)

# OpenCV
add_subdirectory("../../extern/opencv")


find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
