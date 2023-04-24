#include "aux.h"

    
extern int row_size;
/* 
* Automata rules for cells that survive or die
*/
void newValue( int *lower, int *value, int *upper, int *tmp ) {
    int i;
    int neighbor[row_size];
    
    for(i = 0; i < row_size; i++) {
        neighbor[i] = upper[ LEFT(i, row_size) ] + upper[i] + upper[ RIGHT(i, row_size) ]
                        + value[ LEFT(i, row_size) ] + value[ RIGHT(i, row_size) ]	
                        + lower[ LEFT(i, row_size) ] + lower[i] + lower[ RIGHT(i, row_size) ];
    
        /* If a cell is off and has 3 living neighbors, it will become 
        * alive in the next generation. */
        if ( value[i] == 0 ) {
            if ( neighbor[i] == 3 ) {
                tmp[i] = 1;
            }
            else {
                tmp[i] = 0;
            }
        }
        /* If a cell is on and has 2 or 3 living neighbors, it survives;
        * otherwise, it dies in the next generation.*/
        else {
            if ( neighbor[i] >= 2 && neighbor[i] <= 3 ) {
                tmp[i] = 1; 
            }
            else {
                tmp[i] = 0;
            }
        }
    }	

}

/* 
* Get the index row from grids.
*/
void getRow( int index, int *grid, int *row ) {
    int i;
    for ( i = 0; i < row_size; i++ ) {
        row[i] = grid[ index * row_size + i ];
    }
}

/*
* Copy a row.
*/
void copyRow( int *srcRow, int *desRow ) {
    int i;
    for ( i = 0; i < row_size; i++ ) {
        desRow[i] = srcRow[i];
    }
}

/*
* Util function for printing a row. Used for debugging.
*/
void printRow( int *row ) {
    int j;
    for ( j = 0; j < row_size; j++ ) {
    printf ( " %d ", row[j] );
    }
    printf( " \n" );
} 

void bitmap(int *recv_buf, int total_entries, int row_size, int nprocs, int i){
    
    char buffig[1024];
    snprintf(buffig, sizeof buffig, "bitmap_%i_%i_%i.bit", i, row_size, nprocs);
    FILE *outmnfig = fopen(buffig, "wb");
    if (outmnfig == NULL){
        printf("Could not open writing file.");
    }
    
    unsigned char *img = NULL;
    int filesize = 54 + 3 * row_size * row_size;  //w is your image width, h is image height, both int
    img = (unsigned char *) malloc(3 * row_size * row_size);
    memset(img, 0, 3 * row_size * row_size);    
    
    for (int i = 0; i < total_entries; i++) {
        if (recv_buf[i] == 1){
            img[(i)*3+2] = (unsigned char)(0);
            img[(i)*3+1] = (unsigned char)(0);
            img[(i)*3+0] = (unsigned char)(0);
        }
        else {
            img[(i)*3+2] = (unsigned char)(255);
            img[(i)*3+1] = (unsigned char)(255);
            img[(i)*3+0] = (unsigned char)(255);
        }
    }


    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       row_size    );
    bmpinfoheader[ 5] = (unsigned char)(       row_size>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       row_size>>16);
    bmpinfoheader[ 7] = (unsigned char)(       row_size>>24);
    bmpinfoheader[ 8] = (unsigned char)(       row_size    );
    bmpinfoheader[ 9] = (unsigned char)(       row_size>> 8);
    bmpinfoheader[10] = (unsigned char)(       row_size>>16);
    bmpinfoheader[11] = (unsigned char)(       row_size>>24);


    fwrite(bmpfileheader,1,14,outmnfig);
    fwrite(bmpinfoheader,1,40,outmnfig);
    for(int i=0; i<row_size; i++)
    {
        fwrite(img+(row_size*(row_size-i-1)*3),3,row_size,outmnfig);
        fwrite(bmppad,1,(4-(row_size*3)%4)%4,outmnfig);
    }

    free(img);
    fclose(outmnfig);
}


int *parallel_read(char *input_file, int *proc_size, int rank, int nprocs){
    
    int *a;
    MPI_File my_file;
    MPI_Offset filesize;
    MPI_Status status;
    
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &my_file);
    MPI_File_get_size(my_file, &filesize);
    
    filesize = filesize / sizeof(int);
    (*proc_size) = filesize / nprocs;
    a = malloc((*proc_size) * sizeof(int));
    
    MPI_File_set_view(my_file, rank * (*proc_size) * sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
    MPI_File_read(my_file, a, (*proc_size), MPI_INT, &status);
    
    MPI_File_close(&my_file);
    
    return a;
}
