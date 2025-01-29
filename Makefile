CC = nvcc
CFLAGS = -arch=sm_60 -std=c++11 -O2
SRC_DIR = env
OBJ_DIR = build
TARGET = main

$(shell mkdir -p $(OBJ_DIR))

all: $(TARGET)

$(TARGET): $(OBJ_DIR)/env.o main.o
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ_DIR)/env.o main.o

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu -o main.o

$(OBJ_DIR)/env.o: $(SRC_DIR)/env.cu $(SRC_DIR)/env.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/env.cu -o $(OBJ_DIR)/env.o

clean:
	rm -f $(OBJ_DIR)/*.o main.o $(TARGET)
