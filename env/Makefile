CC = nvcc
CFLAGS = -arch=sm_89 -std=c++17 -O2
OBJ_DIR = ../build

$(shell mkdir -p $(OBJ_DIR))

all: $(OBJ_DIR)/env.o

$(OBJ_DIR)/env.o: env.cu env.h
	$(CC) $(CFLAGS) -c env.cu -o $(OBJ_DIR)/env.o

clean:
	rm -f $(OBJ_DIR)/env.o
