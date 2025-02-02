# Compiler and flags
CC = nvcc
CFLAGS = -arch=sm_89 -std=c++17 -O2

# Directories
OBJ_DIR = ../build
# Use immediate assignment with strip to avoid trailing spaces
# LIBTORCH_PATH := $(strip ../libtorch)

# Include directories for LibTorch (verify by echoing if necessary)
# INCLUDE_DIRS = -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include

# Library directories and LibTorch libs (for linking in the top-level Makefile)
# LIBRARY_DIRS = -L$(LIBTORCH_PATH)/lib
# LIBTORCH_LIBS = -ltorch -lcaffe2 -lc10

# Ensure output directory exists
$(shell mkdir -p $(OBJ_DIR))

# Targets for compiling network.o and ppo.o
all: $(OBJ_DIR)/network.o $(OBJ_DIR)/ppo.o

# Rule to compile network.cu
$(OBJ_DIR)/network.o: network.cu network.h
	$(CC) $(CFLAGS) -c network.cu -o $(OBJ_DIR)/network.o

# Rule to compile ppo.cu
$(OBJ_DIR)/ppo.o: ppo.cu ppo.h ../env/env.h
	$(CC) $(CFLAGS) -c ppo.cu -o $(OBJ_DIR)/ppo.o

# Clean up build files
clean:
	rm -f $(OBJ_DIR)/network.o $(OBJ_DIR)/ppo.o
