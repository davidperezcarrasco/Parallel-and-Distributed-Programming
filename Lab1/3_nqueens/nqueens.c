#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

struct timeval  stop,
                init;

static inline __attribute__((always_inline)) void start_timer(){gettimeofday(&init, NULL);}

static inline __attribute__((always_inline)) void stop_timer(long long *elp){
    gettimeofday(&stop, NULL);
    *elp = (stop.tv_sec - init.tv_sec) * 1e6 + stop.tv_usec - init.tv_usec;}
    
long long itertime;
long long tottime;
        
        
int Fitness(int *array, int size){
    int j, k;
    int attack = 0;

    #pragma omp parallel for reduction(+:attack) private(k)
    for (j = 0; j < size; j++){
        for (k = (size-1); k > j; k--){
            if ((array)[k] == (array)[j] || (((abs((array)[k]-(array)[j])) == (k-j)))) {
                attack++;
            }
        }
    }
    return attack;
}      


void Mutation(int *parent, int size) // Mutation step. HEURISTIC MUTATION
{
    int fit1 = Fitness(parent, size);
    int fit2;
    int pos1, pos = rand()%size;
    int oldcol = (parent)[pos];
    int newcol = oldcol;
    int i;
    int mut = rand()%100;

    if (mut < 49){
        for (i = 0; i < size; i++){
            (parent)[pos] = i; fit2 = Fitness(parent, size);
            if (fit2 <= fit1) {newcol = i; fit1 = fit2;} else ;
        }
        (parent)[pos] = newcol;
    }
    else{
        pos1 = rand()%size;
        newcol = (parent)[pos1]; (parent)[pos1] = (parent)[pos]; (parent)[pos] = newcol;
        fit2 = Fitness(parent, size);
        if (fit1 > fit2) {
            newcol = (parent)[pos1]; 
            (parent)[pos1] = (parent)[pos]; 
            (parent)[pos] = newcol;
        }
    }
}

void OnePointCrossover(int *parent1, int *parent2, int size){
    int crossing = rand()%(size-1);
    int aux;
    for (int i = crossing; i < size; i++){
        aux = (parent2)[i]; 
        (parent2)[i] = (parent1)[i]; 
        (parent1)[i] = aux;
    }
}

int Tournament(int *individual_fitness, int tsize, int population){
    int aux;
    int index = rand()%population;

    for (int i = 0; i < tsize; i++){
        aux = rand()%population;
        if ((individual_fitness)[aux] < (individual_fitness)[index]) index = aux; else ;
    }
    return index;
}
        
        
int population_fitness(int **individuals, int *individual_fitness, int *bestfitness_ind, int population, int size, int *best_ind){
    int minfit = 1E4;
    
    for (int i = 0; i < population; i++){
        (individual_fitness)[i] = Fitness(individuals[i], size);
        if ((individual_fitness)[i] < minfit) {
            (bestfitness_ind) = (individuals)[i]; 
            minfit = (individual_fitness)[i];
            *best_ind = i;
        }
        if (minfit == 0) {break;}
    }
    return minfit;
}
        
void create_population(int **individuals, int **newgen, int population, int size){
    for (int i = 0; i < population; i++){
        (individuals)[i]   = malloc(sizeof(int)*size);
        (newgen)[i]        = malloc(sizeof(int)*size);
        for(int j = 0; j < size; j++){
            (individuals)[i][j] = rand()%size;
        } 
    }
}

void copy_pop(int **individuals, int **newgen, int population, int size){
    for (int i = 0; i < population; i++){
        for (int j = 0; j < size; j++){
            individuals[i][j] = newgen[i][j];
        }
    }
}
        

int main(int argc, char *argv[])
{
    srand(1);
    int size, iter_max, population;
    
    if (argc < 4 
        || sscanf(argv[1], "%d", &size)!=1  
        || sscanf(argv[2], "%d", &population)!=1
        || sscanf(argv[3], "%d", &iter_max)!=1)
            {printf("Please add as arguments when executing the program the number of queens (as integer), the total population (as integer multiple of 5) and the maximum number of iterations (as integer).\n"); return -1;}
            
    printf("\nN-Queens problem. Number of queens = %i\n", size); 
    
    int tsize = population/5;
    
    int solved                  = -1;
    int iter                    = 0;
    int t1, t2;
    int minfit                  = 0;
    int **individuals           = malloc(sizeof(int *)*population);    // Stores solutions
    int **newgen                = malloc(sizeof(int *)*population);    // Stores children
    int *individual_fitness     = malloc(sizeof(int)*population);      // Fitness ind
    int *parent1                = malloc(sizeof(int)*size);            // Parent 1
    int *parent2                = malloc(sizeof(int)*size);            // Parent 2
    int *bestfitness_ind        = malloc(sizeof(int)*size);            // Best Fitness
    int best_ind                = 0;
    /* Create population of solutions */
    create_population(individuals, newgen, population, size);
    
    tottime = 0;
    /* Iterate until a solution is found or iter_max is reached */
    while (solved == -1 && iter < iter_max)
    {
        start_timer();
        
        minfit = population_fitness(individuals, individual_fitness, bestfitness_ind, population, size, &best_ind);
        if (minfit == 0) {solved = 1; break;}
        
        for (int i = 0; i < population/2; i++)
        {
            t1 = t2;
            while (t1 == t2){
                t1 = Tournament(individual_fitness, tsize, population);
                t2 = Tournament(individual_fitness, tsize, population);
            }
            
            for (int j = 0; j < size; j++)
            {
                parent1[j] = individuals[t1][j]; 
                parent2[j] = individuals[t2][j];
            }

            OnePointCrossover(parent1, parent2, size);
            
            Mutation(parent1, size); 
            Mutation(parent2, size);
            
            for (int j = 0; j < size; j++){
                newgen[2*i][j] = parent1[j]; 
                newgen[2*i+1][j] = parent2[j];
            }
        }
        
        copy_pop(individuals, newgen, population, size);
        
        iter ++;
        
        stop_timer(&itertime);
        tottime += itertime;
    }
    
    if (solved == -1){printf("\nNo solution was found for %i queens in %i iterations.\n\n", size, iter_max);}
    else if (solved > -1){
        printf("\nProblem Solved: Fitness = %i. Iteration: %i. Total Time: %lf\n\n", individual_fitness[best_ind], (iter), (double) tottime / 1000000.0); 
        for(int j = 0; j < size; j++){
            printf("(%i,%i)", j, individuals[best_ind][j]);
        }
        printf("\n\n");
    }
    
    return 0;
}

