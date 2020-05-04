.PHONY: sfml console clean


# Choose compiler
CC = g++

console:
	@echo "Compiling..."
	@${CC} -c main.cpp -I include/
	@${CC} -c src/node.cpp -I include/
	@${CC} -c src/topology.cpp -I include/
	@g++ main.o net.o node.o topology.o -o app.o
	@echo "Done"
	@echo "------------------------"
	@./app.o
	@echo "Or, if you want to use whole functionality, install SFML"
	@echo "And then use make all instead just make"

all:
	@echo "Compiling..."
	@${CC} -c main.cpp -I include/
	@${CC} -c src/node.cpp -I include/
	@${CC} -c src/topology.cpp -I include/
	@${CC} -c src/plot.cpp src/net.cpp -I include/
	@g++ main.o net.o node.o topology.o plot.o -o app.o -lsfml-graphics -lsfml-window -lsfml-system
	@echo "Done"
	@echo "------------------------"
	@./app.o

clean:
	@echo "Cleaning up..."
	rm *.o
