TARGET  := trans
OBJ := trans.o

INCLUDES    := -I /usr/local/cuda/include
LIBRARY_PATH    := -L /usr/lib64/
LIB     := -lOpenCL

all: $(TARGET)

$(TARGET): $(OBJ)
	g++ $(OBJ) -o $(TARGET) $(LIB)

$(OBJ): $(TARGET).cpp
	g++ -c $(TARGET).cpp $(INCLUDES) $(LIBRARY_PATH)

clean:
	rm *o $(TARGET) *ptx
