.PHONY: all clean


# Choose compiler
CC = g++

all:
	@echo "Compiling..."
	@${CC} main.cpp src/net.cpp src/node.cpp src/topology.cpp -o start.o -I include/
	@echo "Done"
	@echo "------------------------"
	@./start.o


clean:
	@echo "Cleaning up..."
	rm *.o
