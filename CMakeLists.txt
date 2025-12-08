cmake_minimum_required(VERSION 3.15)
project(YOLOv8_CPP VERSION 0.1
	DESCRIPTION ""
	LANGUAGES C CXX CUDA)


find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)

# Project subdirectories
add_subdirectory("project/v1_serial")
add_subdirectory("project/v1_serial_poc")
add_subdirectory("project/v2_MPI_only")
