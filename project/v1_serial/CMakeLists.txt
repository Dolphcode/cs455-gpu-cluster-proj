add_executable(YOLOv8_CPP_Serial_Main
	src/main.cpp
	src/prepostproc.cpp
	src/layer_structs.cpp
)

target_link_libraries(YOLOv8_CPP_Serial_Main
	PRIVATE
		${OpenCV_LIBS}
)

target_include_directories(YOLOv8_CPP_Serial_Main
	PRIVATE
		include
)
