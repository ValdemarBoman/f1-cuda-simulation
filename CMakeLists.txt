cmake_minimum_required(VERSION 3.17)
project(F1_CUDA_Simulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")  # DGX A100 = sm_80

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src)

# Neural network library
add_library(sac_cuda STATIC
    src/sac/network.cu
    src/sac/sac_cuda.cu
)
target_link_libraries(sac_cuda ${CUDA_LIBRARIES} cublas curand)

# Main executable
add_executable(f1_simulation src/main.cpp)
target_link_libraries(f1_simulation sac_cuda ${CUDA_LIBRARIES} cublas curand)