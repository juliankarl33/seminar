CXXFLAGS =	-O0 -g -Wall -std=c++11 -fmessage-length=0

OBJS =		seminar.o

LIBS =

TARGET =	seminar

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
