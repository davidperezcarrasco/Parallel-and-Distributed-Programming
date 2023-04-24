CC=pgcc
OBJ=imgproc
FLAGS= -fast -Minfo=all  

all: $(OBJ).c
	$(CC) $(FLAGS) -o $(OBJ) $<
clean:
	rm $(OBJ)
cleanall:
	rm $(OBJ) *.err *.out
gpu: $(OBJ).c
	$(CC) $(FLAGS) -ta=tesla -o $(OBJ) $<
cpu: $(OBJ).c
	$(CC) $(FLAGS) -ta=multicore -o $(OBJ) $<
