#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>


int* a; int* b;

int dot(int mode, int n_threads, int size){

  int result = 0;
  if (mode==1){
    for (int i=0;i<size;i++){
      result += a[i]*b[i];
    }
    
  } else if (mode==2){
    
    
    #pragma omp parallel for reduction(+:result) num_threads(n_threads)
    
    for (int i=0;i<size;i++){
      result += a[i]*b[i];
    }
    
  } else if (mode ==3){
    #pragma omp parallel for simd reduction(+:result) num_threads(n_threads)
    
    for (int i=0;i<size;i++){
      result += a[i]*b[i];
    }
  }
  return result;
}

int main(int argc, char* argv[]) {
  
  if (argc < 3){
    printf("Not enough arguments given (you need to provide 3 integers).");
    _exit(0);
  }
  
  int mode = atoi(argv[1]);
  int n_threads = atoi(argv[2]);
  int size = atoi(argv[3]);

  a = malloc(size*sizeof(int));
  b = malloc(size*sizeof(int));

  for(int i = 0 ; i < size ; i ++){
    a[i] = (int) (rand() %size -size/2);
    b[i] = (int) (rand() %size -size/2);
  }
  
  srand(1);

  #ifdef _OPENMP
  double start = omp_get_wtime();
  
  int sum = dot(mode,n_threads,size);

  double end = omp_get_wtime();
 
  printf("%.4e\t%i\n",  end-start, sum);
  #endif
  
  return 0;
}
