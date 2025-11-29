add_executable(YOLOv8_CPP_Serial
	src/main.cpp
	src/layer_structs.cpp
)

target_include_directories(YOLOv8_CPP_Serial
	PRIVATE	
		include
)
