CC=g++
CFLAGS=-Wall -O3 -ansi -fopenmp
LDFLAGS= 
SOURCES=mandelbrot.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=mandelbrot
LIBS=-lSDL -lm -lgomp

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(LIBS) $(OBJECTS) -o $@

.cc.o:
	$(CC) $(CFLAGS) -c $< -o $@



clean:
	rm -f *.o $(EXECUTABLE)
