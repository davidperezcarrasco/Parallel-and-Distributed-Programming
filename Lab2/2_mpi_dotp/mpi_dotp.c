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
    double* vector_p;
    double* vector_q;
    double start_time, final_time;

    //initialize MPI environment
    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    //initialize vector chunks
    char * path_p = "/shared/Labs/Lab_2/array_p.bin";
    char * path_q = "/shared/Labs/Lab_2/array_q.bin";
    int p_size1 = 0;
    int p_size2 = 0;
    vector_p = par_read(path_p, &p_size1, rank, size);
    vector_q = par_read(path_q, &p_size2, rank, size);
    
    double result = 0;
    
    //we further paralelize using OpenMP, concretely, since it is the repetition of one instruction and multiple data (simd)
    #pragma omp parallel for simd reduction(+:result)
    for (int i= 0; i < p_size1; i++){
        result += vector_p[i] * vector_q[i];
    }

    //cobine all private results
    double result_final = 0;
    MPI_Reduce(&result, &result_final, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
    if (rank == 0){
	    printf("\nNumber of Processes: %d\n", size);
	    final_time = MPI_Wtime();
        printf("\nTotal time = %lf\n\n", final_time - start_time);
	    printf("result = %lf\n\n", result_final); 
    } 
    MPI_Finalize();
    return 0;
}
