CC = nvcc
CFLAGS = -arch=sm_60 -std=c++11 -O2
TARGET = env
OBJ_DIR = ../build

$(shell mkdir -p $(OBJ_DIR))

all: $(OBJ_DIR)/$(TARGET)

$(OBJ_DIR)/$(TARGET): env.o
	$(CC) $(CFLAGS) -o $(OBJ_DIR)/$(TARGET) env.o

env.o: env.cu env.h
	$(CC) $(CFLAGS) -c env.cu -o env.o

clean:
	rm -f env.o $(OBJ_DIR)/$(TARGET)
