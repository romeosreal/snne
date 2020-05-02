.PHONY: all clean


# Choose compiler
CC = g++

all:
	@echo "Compiling..."
	@${CC} main.cpp src/onn.cpp -o start.o -I include/
	@echo "Done"
	@echo "------------------------"
	@./start.o


clean:
	@echo "Cleaning up..."
	rm *.o
