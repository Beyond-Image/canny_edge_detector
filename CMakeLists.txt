#Specify the version
cmake_minimum_required(VERSION 3.29)

#Name your project here
project(canny_edge_detector LANGUAGES CXX CUDA)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(Boost_DIR "C:/Users/Felipe/vcpkg/installed/x86-windows/share")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


find_package(TIRA REQUIRED)
find_package(glm REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)


#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
else()
	find_package(X11 REQUIRED)
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ( MSVC )

#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${TIRA_INCLUDE_DIRS}
			${CUDA_INCLUDE_DIRS}
			${X11_INCLUDE_DIR}
			${CMAKE_SOURCE_DIR}/include
)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)


#create an executable
add_executable(canny_edge_detector
		canny_edge_detector.cpp
		canny_edge_detector.cu
		gpu_kernels.h
)

target_link_libraries(canny_edge_detector
			glm::glm
			Boost::program_options
			cuda
)