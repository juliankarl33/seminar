CXXFLAGS =	-O3 -fopenmp -g  -Werror -Winline -pedantic -Wall -std=c++11 -fmessage-length=0

OBJS =		seminar.o

LIBS = -fopenmp

TARGET =	seminar

$(TARGET):	$(OBJS) Makefile
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
