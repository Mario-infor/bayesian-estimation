
CXXFLAGS = -Wall -O3 -I ./

all: SegColorLab 

SegColorLab: SegColorLab.o Circle.o
	g++ $(CXXFLAGS) -o SegColorLab SegColorLab.o Circle.o `pkg-config opencv4 --libs`

SegColorLab.o: SegColorLab.cpp
	g++ $(CXXFLAGS) -c SegColorLab.cpp `pkg-config opencv4 --cflags`

Circle.o: Circle.cpp
	g++ $(CXXFLAGS) -c Circle.cpp `pkg-config opencv4 --cflags`

clean: 
	rm *.o SegColorLab 
