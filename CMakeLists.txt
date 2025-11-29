cmake_minimum_required(VERSION 3.15)
project(YOLOv8_CPP VERSION 0.1
	DESCRIPTION ""
	LANGUAGES C CXX CUDA)


find_package(OpenMP REQUIRED)
#find_package(OpenCV REQUIRED)

# Project subdirectories
add_subdirectory("project/v1_serial")
