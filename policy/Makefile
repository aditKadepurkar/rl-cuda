CC = nvcc
CFLAGS = -arch=sm_60 -std=c++11 -O2
OBJ_DIR = ../build
# CUDA_LIBS = -lcublas -lcudnn -lcurand

$(shell mkdir -p $(OBJ_DIR))

all: $(OBJ_DIR)/network.o $(OBJ_DIR)/ppo.o

$(OBJ_DIR)/network.o: network.cu network.h
	$(CC) $(CFLAGS) -c network.cu -o $(OBJ_DIR)/network.o

$(OBJ_DIR)/ppo.o: ppo.cu ppo.h ../env/env.h
	$(CC) $(CFLAGS) -c ppo.cu -o $(OBJ_DIR)/ppo.o

clean:
	rm -f $(OBJ_DIR)/network.o $(OBJ_DIR)/ppo.o
