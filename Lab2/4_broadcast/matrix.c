#include <unistd.h>
#include <stdio.h>
#include "mpi.h"
#include "unistd.h"
#include <stdlib.h>


int *new_matrix(int size, int rank){
    int *a = (int *)malloc(size*size*sizeof(int*));
    int offset;
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            offset = i*size+j; //The matrix i,j position relates to i*num_of_columns+j as all the rows are concatenated 
            if (i==j) a[offset]=rank;
            else a[offset] = 0;
        }
    }

    return a;
}


void print_matrix(int *matrix, int size){
    int offset;
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            offset = i*size+j;
            printf("%d ",matrix[offset]);
        }
        printf("\n");
    }
}


int main(int argc, char **argv){
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *matrix = new_matrix(size,rank);
    
    if (rank==0) {
        printf("Initial Matrix (rank 0) \n");
        print_matrix(matrix,size);
    }

    MPI_Datatype diagonal;
    MPI_Type_vector(size,1,size+1,MPI_INT,&diagonal); //In order to get the riagonal we select a stride of size+1
    MPI_Type_commit(&diagonal);

    MPI_Gather(matrix,1,diagonal,matrix,size,MPI_INT,0,MPI_COMM_WORLD);
    
    if (rank==0) {
        printf("\nFinal Matrix (rank 0) \n");
        print_matrix(matrix,size);
    }

    MPI_Type_free(&diagonal);

    MPI_Finalize();
    return 0;
    
}
 