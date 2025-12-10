add_executable(YOLOv8_CPP_CUDA_MPI
	src/main.cpp
	src/layer_structs.cpp
	src/model.cu
	src/preprocess.cpp
)

set_target_properties(YOLOv8_CPP_CUDA_MPI
	PROPERTIES
	CUDA_ARCHITECTURES "50"
)

target_link_libraries(YOLOv8_CPP_CUDA_MPI
	PRIVATE
		${OpenCV_LIBS}
		MPI::MPI_CXX
		${CUDA_LIBRARIES}
)

target_include_directories(YOLOv8_CPP_CUDA_MPI
	PRIVATE
		include
	PUBLIC
		${MPI_INCLUDE_PATH}
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
