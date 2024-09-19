cmake_minimum_required(VERSION 3.10)
project(CUDA_Winograd LANGUAGES CXX CUDA)

# Set the standard for C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
find_package(CUDA REQUIRED)

# Add source files
set(SOURCES
    main.cu
    Kernel128_winograd.cu
    Kernel256_winograd.cu
    Kernel512_winograd.cu
    util.c
)

# Specify the executable
add_executable(Test ${SOURCES})

# Set the architecture (modify according to your GPU)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_61")

# Link CUDA libraries
target_link_libraries(Test ${CUDA_LIBRARIES})

# Set compile options
set_target_properties(Test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"  # Modify based on your GPU
)

# Specify directories for includes (if needed)
include_directories(${CMAKE_SOURCE_DIR}/include)
