CXXFLAGS =	-O3 -fopenmp -g  -Werror -Winline -pedantic -Wall -std=c++11 -fmessage-length=0

OBJS =		mgsolve.o

LIBS = -fopenmp

TARGET =	mgsolve

$(TARGET):	$(OBJS) Makefile
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
