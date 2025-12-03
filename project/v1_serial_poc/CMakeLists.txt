add_executable(YOLOv8_CPP_Serial
	src/main.cpp
	src/layer_structs.cpp
	src/serial_funcs.cpp
)

target_link_libraries(YOLOv8_CPP_Serial
	PRIVATE
		${OpenCV_LIBS}
)

target_include_directories(YOLOv8_CPP_Serial
	PRIVATE	
		include
)
