COMPILER	= gcc
CFLAGS		= -std=c99 -std=gnu11 -Wall -march=native -fopenmp -fopt-info-vec-{optimized,missed} -O3
LIBS		= -lm
OBJ		= dotp
TARGET		= a.out

.PHONY: all

all:
	$(COMPILER) $(CFLAGS) $(OBJ).c $(LIBS) -o $(TARGET)
	
	
.PHONY: clean
	
clean:
	rm -f $(TARGET)
