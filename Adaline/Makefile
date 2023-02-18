CC = g++ -pthread
CFLAGS = -I -g -w -Wall -std=c++14
BINARY_NAME = adaline

main: src/main.cpp src/adaline.cpp src/csvHandling.cpp
	$(CC) $(CFLAGS) -o $(BINARY_NAME)  src/main.cpp src/adaline.cpp src/csvHandling.cpp

clean:
	rm $(BINARY_NAME)
