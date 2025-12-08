add_executable(YOLOv8_CPP_MPI
	src/main.cpp
	src/prepostproc.cpp
	src/layer_structs.cpp
	src/serial_funcs.cpp
)

target_link_libraries(YOLOv8_CPP_MPI
	PRIVATE
		${OpenCV_LIBS}
		MPI::MPI_CXX
)

target_include_directories(YOLOv8_CPP_MPI
	PRIVATE
		include
	PUBLIC
		${MPI_INCLUDE_PATH}
)
