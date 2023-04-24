#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"


double * par_read(char * in_file, int * p_size, int rank, int nprocs){
    //open file
    MPI_File fh;
    MPI_Offset total_size;
    int ferr = MPI_File_open(MPI_COMM_WORLD, in_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ferr){
        MPI_Finalize();
        exit(1);
    }
    
    //we divide into corresponding chunks and read all at once
    MPI_File_get_size(fh, &total_size);
    int double_size;
    MPI_Type_size(MPI_DOUBLE, &double_size);
    int elems = (total_size / nprocs) / double_size;
    MPI_Offset offset = elems * rank * double_size;
    double* vector = malloc(double_size * elems);
    MPI_File_read_at(fh, offset, vector, elems, MPI_DOUBLE, MPI_STATUS_IGNORE);
    *p_size = elems; 
    MPI_File_close(&fh);
    return vector;

}

int main(int argc, char ** argv){
    int rank, size;
    double* matrix;
    double* vector;
    double start_time, final_time;

    //initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    //divide the vector into independent chunks
    char * path_matrix = "/shared/Labs/Lab_2/matrix.bin";
    char * path_vector = "/shared/Labs/Lab_2/matrix_vector.bin";
    int matrix_size = 0;
    int vector_size = 0;
    matrix = par_read(path_matrix, &matrix_size, rank, size);
    vector = par_read(path_vector, &vector_size, rank, size);
   
    //get the vector size, which is also the size of each matrix row
    int total_vector_size = vector_size * size;
    
    //gather all the vector elements
    double vector_b[total_vector_size];
    MPI_Allgather(vector, vector_size, MPI_DOUBLE, vector_b, vector_size, MPI_DOUBLE, MPI_COMM_WORLD);
    //compute product
	//get the initial index for the vector
    int partial_size = matrix_size / total_vector_size;
    double result[partial_size];
    start_time = MPI_Wtime();
    for (int i= 0; i < matrix_size; i++){
        result[i / total_vector_size] += vector_b[i % total_vector_size] * matrix[i];
    }
    final_time = MPI_Wtime();
    //cobine all private results
    double result_final[total_vector_size];
    MPI_Gather(result, partial_size, MPI_DOUBLE, result_final, partial_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0){
        printf("\nTotal time = %lf\n\n", final_time - start_time);
	printf("result = %lf\n\n", result_final[0]); 
    } 
    free(matrix);
    free(vector);

    MPI_Finalize();
    return 0;
}
