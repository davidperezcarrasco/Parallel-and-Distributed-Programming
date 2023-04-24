CC=pgcc
OBJ=imgproc_streams
FLAGS= -fast -Minfo=accel  

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
