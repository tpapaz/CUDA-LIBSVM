cmake_minimum_required(VERSION 3.12)

project(gpu-libsvm LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
# This will find the toolkit and set up the necessary variables.
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (e.g., 75 for Turing)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add the executable target
add_executable(svm-train-gpu src/svm.cpp src/svm-train.c)

# Explicitly tell CMake to compile the .c file with nvcc
set_source_files_properties(src/svm-train.c PROPERTIES LANGUAGE CUDA)

# Add include directories
target_include_directories(svm-train-gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link against the modern CUDA targets
target_link_libraries(svm-train-gpu PRIVATE CUDA::cublas CUDA::cudart)
