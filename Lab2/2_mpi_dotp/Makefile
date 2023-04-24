DIR 	:= ${CURDIR}

CFLAGS		= -std=c99 -std=gnu11 -Wall -fopenmp
LIBS		= -lm
COMPFLAGS	= -O3

COMPILER = mpicc

SRCS	:= $(wildcard *.c)
DEPS	:= $(wildcard *.h)
OBJ	:= $(patsubst %.c, %.o, $(SRCS))

TARGET      = a.out

.PHONY: all

all: $(TARGET)

$(OBJ): %.o: %.c
	$(COMPILER) -c $< -o $@ $(CFLAGS) $(COMPFLAGS) 


$(TARGET): $(OBJ)
	$(COMPILER) -o $(TARGET) $(OBJ) $(LIBS) $(CFLAGS) $(COMPFLAGS)


.PHONY: clean

clean:
	rm -f *.o a.out
