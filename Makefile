CC = nvcc
CFLAGS = -arch=sm_89 -std=c++11 -O2
SRC_DIRS = env policy
OBJ_DIR = build
TARGET = main
CUDA_LIBS = -lcublas -lcudnn -lcurand

$(shell mkdir -p $(OBJ_DIR))

all: $(TARGET)

$(TARGET): $(OBJ_DIR)/env.o $(OBJ_DIR)/network.o $(OBJ_DIR)/ppo.o main.o
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ_DIR)/env.o $(OBJ_DIR)/network.o $(OBJ_DIR)/ppo.o main.o $(CUDA_LIBS)

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu -o main.o

$(OBJ_DIR)/env.o:
	$(MAKE) -C env

$(OBJ_DIR)/network.o $(OBJ_DIR)/ppo.o:
	$(MAKE) -C policy

clean:
	rm -f $(OBJ_DIR)/*.o main.o $(TARGET)
	$(MAKE) -C env clean
	$(MAKE) -C policy clean
