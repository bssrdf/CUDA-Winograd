cmake_minimum_required(VERSION 3.12)  # CMake version needs to support CUDA

# Project name
project(CUDA_Winograd LANGUAGES CXX CUDA)

# Specify the C++ standard (if required)
set(CMAKE_CXX_STANDARD 11)

# Find CUDA package
find_package(CUDA REQUIRED)

# Include directories (adjust path if necessary)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files (adjust filenames if necessary)
set(SOURCES
    src/winograd.cu
    src/main.cpp
)

# Set target executable
add_executable(CUDA_Winograd ${SOURCES})

# Link against CUDA
target_link_libraries(CUDA_Winograd ${CUDA_LIBRARIES})

# Set CUDA-specific properties if needed
set_target_properties(CUDA_Winograd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include any additional flags you need for your project (adjust as necessary)
