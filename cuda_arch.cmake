set(CMAKE_CUDA_ARCH_BIN "60;61;70;75;80")  # Specify the CUDA architectures to target
set(CMAKE_CUDA_ARCH_PTX "60;61;70;75;80")  # Specify the CUDA architectures for PTX generation

# Optionally, you can set the minimum required CUDA version
set(CMAKE_CUDA_COMPILER_VERSION "11.0")  # Adjust this to your installed CUDA version

# Enable CUDA's unified memory support
set(CMAKE_CUDA_UNIFIED_MEMORY "ON") 

# Set the CUDA standard to C++14 or higher
set(CMAKE_CUDA_STANDARD 14)  # Adjust this as needed

# Additional flags for optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")  # Enable optimizations and fast math

# Include directories for CUDA
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/simulation)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/sac)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/parallel)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/utils)