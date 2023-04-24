COMPILER	= gcc
CFLAGS		= -std=c99 -std=gnu11 -Wall -O3 -fopenmp
LIBS		= -lm
OBJ		= quicksort
TARGET		= a.out

.PHONY: all

all:
	$(COMPILER) $(CFLAGS) $(OBJ).c $(LIBS) -o $(TARGET)
	
	
.PHONY: clean
	
clean:
	rm -f $(TARGET)
