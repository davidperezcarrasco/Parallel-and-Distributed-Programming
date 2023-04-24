DIR := ${CURDIR}

CFLAGS = -std=c99 -std=gnu11 -Wall -fopenmp
LIBS = -lm
COMPFLAGS = 

COMPILER = mpicc

SRCS	:= $(wildcard *.c)
DEPS	:= $(wildcard *.h)
OBJ	:= $(patsubst %.c, %.o, $(SRCS))

TARGET      = a.out

.PHONY: all

all: $(TARGET)

$(OBJ): %.o: %.c
	$(COMPILER) $(CFLAGS) $(COMPFLAGS) $(LIBS) -c $< -o $@ 


$(TARGET): $(OBJ)
	$(COMPILER) $(CFLAGS) $(COMPFLAGS) $(LIBS) -o $(TARGET) $(OBJ)


.PHONY: clean

clean:
	rm -f *.o a.out
