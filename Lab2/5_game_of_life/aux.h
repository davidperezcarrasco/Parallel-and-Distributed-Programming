
#ifndef aux_h_
    #define aux_h_
    
    
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <mpi.h>
#include <string.h>


#define LEFT(i, size)   ((size + (i) - 1) % size)
#define RIGHT(i, size)  ((size + (i) + 1) % size)


/* 
* Automata rules for cells that survive or die
*/
void newValue( int *lower, int *value, int *upper, int *tmp );
/* 
* Get the index row from grids.
*/
void getRow( int index, int *grid, int *row );
/*
* Copy a row.
*/
void copyRow( int *srcRow, int *desRow );
/*
* Util function for printing a row. Used for debugging.
*/
void printRow( int *row );

int *parallel_read(char *input_file, int *proc_size, int rank, int nprocs);

void bitmap(int *recv_buf, int total_entries, int row_size, int nprocs, int i);

#endif 
