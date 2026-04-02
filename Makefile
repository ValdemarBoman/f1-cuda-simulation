# Makefile for F1 CUDA Simulation

# Compiler and flags
CXX = nvcc
CXXFLAGS = -O2 -std=c++17 -arch=sm_60

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
MAIN_SRC = $(SRC_DIR)/main.cpp
PHYSICS_SRC = $(SRC_DIR)/simulation/physics.cu
TRACK_SRC = $(SRC_DIR)/simulation/track.cu
CAR_SRC = $(SRC_DIR)/simulation/car.cu
SAC_SRC = $(SRC_DIR)/sac/sac_cuda.cu $(SRC_DIR)/sac/network.cu
EPISODE_MANAGER_SRC = $(SRC_DIR)/parallel/episode_manager.cu
BATCH_PROCESSOR_SRC = $(SRC_DIR)/parallel/batch_processor.cu
CUDA_UTILS_SRC = $(SRC_DIR)/utils/cuda_utils.cu

# Object files
OBJ = $(BUILD_DIR)/main.o $(BUILD_DIR)/physics.o $(BUILD_DIR)/track.o $(BUILD_DIR)/car.o \
      $(BUILD_DIR)/sac_cuda.o $(BUILD_DIR)/network.o $(BUILD_DIR)/episode_manager.o \
      $(BUILD_DIR)/batch_processor.o $(BUILD_DIR)/cuda_utils.o

# Executable
EXEC = $(BIN_DIR)/f1_cuda_simulation

# Rules
all: $(EXEC)

$(EXEC): $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJ) -o $@

$(BUILD_DIR)/main.o: $(MAIN_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/physics.o: $(PHYSICS_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/track.o: $(TRACK_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/car.o: $(CAR_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/sac_cuda.o: $(SRC_DIR)/sac/sac_cuda.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/network.o: $(SRC_DIR)/sac/network.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/episode_manager.o: $(EPISODE_MANAGER_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/batch_processor.o: $(BATCH_PROCESSOR_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/cuda_utils.o: $(CUDA_UTILS_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean