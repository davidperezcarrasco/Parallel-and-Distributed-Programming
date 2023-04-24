#include <unistd.h>
#include <stdio.h>
#include "mpi.h"
#include "unistd.h"

int main(int argc, char **argv){
    int rank, size, split_rank, split_size, even_rank, even_size, odd_rank, odd_size;
    char name[MPI_MAX_OBJECT_NAME], name2[MPI_MAX_OBJECT_NAME];
    
    //PHASE 1
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i=0;i<size;i++){ //We use a for loop in order to get the prints in order
        if ((rank==0) && (i==0)) printf("PHASE 1\n\n");
        if (rank==i) {
            printf("Hi, I'm rank %d. My Communicator is MPI_COMM_WORLD and has a size of %d processes.\n", rank, size);
        }
        MPI_Barrier(MPI_COMM_WORLD); //We wait for all processes 
    }
    
    //PHASE 2
    int color = rank/4;
    
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD,color,rank,&row_comm);
    sprintf(name,"SPLIT_COMM_%d",color);

    MPI_Comm_set_name(row_comm,name);
    int namesize = sizeof(name);
    MPI_Comm_rank(row_comm, &split_rank);
    MPI_Comm_size(row_comm, &split_size);
    
    for (int i=0;i<size;i++){
        if (rank==i) {
            if (i==0) printf("\nPHASE 2\n\n");
            MPI_Comm_get_name(row_comm,name,&namesize); //We obtain the comm name
            printf("Hi, I was rank %d in communicator MPI_COMM_WORLD which had %d processes. Now I'm rank %d in communicator %s which has %d processes.\n", rank, size, split_rank, name, split_size);
        }
        fflush(stdout);
        sleep(1);
        MPI_Barrier(row_comm);
    }

    MPI_Comm_free(&row_comm);
    
    //PHASE 3
    MPI_Comm even_comm, odd_comm;
    MPI_Group group, even_group, odd_group;
    int list[] = {0,2,4,6,8,10,12,14};

    MPI_Comm_group(MPI_COMM_WORLD,&group); //Create the group
    MPI_Group_incl(group,8,list,&even_group); //Include the list ranks (those are the even ones)
    MPI_Comm_create(MPI_COMM_WORLD,even_group,&even_comm);
    sprintf(name2,"EVEN_COMM");
    namesize = sizeof(name2);

    
    if(even_comm != MPI_COMM_NULL){
        MPI_Comm_set_name(even_comm,name2);
        MPI_Comm_size(even_comm,&even_size);
        MPI_Comm_rank(even_comm,&even_rank);

        for (int i=0;i<even_size;i++){
            if (even_rank==i) {
                if (i==0) printf("\nPHASE 3\n\n");
                MPI_Comm_get_name(even_comm,name2,&namesize);
                printf("Hi, I was rank %d in communicator %s which had %d processes. Now I'm rank %d in communicator %s which has %d processes.\n", rank,name, split_size, even_rank, name2, even_size);
            }

            fflush(stdout);
            sleep(1);
            MPI_Barrier(even_comm);
        }
    }
    
    
    if(even_comm != MPI_COMM_NULL) MPI_Comm_free(&even_comm); //Be sure we free the communicator in those processes included 
    MPI_Group_free(&even_group);
    
    
    //PHASE 4
    MPI_Group_excl(group,8,list,&odd_group); //The odd ranks are the ones not in the list
    MPI_Comm_create(MPI_COMM_WORLD,odd_group,&odd_comm);
    sprintf(name2,"ODD_COMM");
    namesize = sizeof(name2);

    
    if(odd_comm != MPI_COMM_NULL){
        MPI_Comm_set_name(odd_comm,name2);
        MPI_Comm_size(odd_comm,&odd_size);
        MPI_Comm_rank(odd_comm,&odd_rank);

        for (int i=0;i<odd_size;i++){
            if (odd_rank==i) {
                if (i==0) printf("\nPHASE 4\n\n");
                MPI_Comm_get_name(odd_comm,name2,&namesize);
                printf("Hi, I was rank %d in communicator MPI_COMM_WORLD which had %d processes. Now I'm rank %d in communicator %s which has %d processes.\n", rank, size, odd_rank, name2, odd_size);
            }

            fflush(stdout);
            sleep(1);
            MPI_Barrier(odd_comm);
          }
    }

    if(odd_comm != MPI_COMM_NULL) MPI_Comm_free(&odd_comm); //Be sure we free the communicator in those processes included 
    MPI_Group_free(&odd_group); //Free both groups, as we could not free the main group before because we used it in PHASE 4
    MPI_Group_free(&group);

    MPI_Finalize();
    return 0;
    
}
 