/*
 *     
 *  IMAGE PROCESSING
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define pixel(i, j, n)  (((j)*(n)) +(i))

/*read*/
void  readimg(char * filename,int nx, int ny, int * image){
  
   FILE *fp=NULL;

   fp = fopen(filename,"r");
   for(int j=0; j<ny; ++j){
      for(int i=0; i<nx; ++i){
         fscanf(fp,"%d", &image[pixel(i,j,nx)]);      
      }
   }
   fclose(fp);
}

/* save */   
void saveimg(char *filename,int nx,int ny,int *image){

   FILE *fp=NULL;
   fp = fopen(filename,"w");
   for(int j=0; j<ny; ++j){
      for(int i=0; i<nx; ++i){
         fprintf(fp,"%d ", image[pixel(i,j,nx)]);      
      }
      fprintf(fp,"\n");
   }
   fclose(fp);

}

/*invert*/
void invert(int* image, int* image_invert, int nx, int ny){
   #pragma acc parallel loop copyin(image[0:nx*ny]) copyout(image_invert[0:nx*ny]) async(1)
   for (int j = 0; j < ny; j++){
      for (int i = 0; i < nx; i++){
         image_invert[pixel(i, j, nx)] = 255 - image[pixel(i, j, nx)];
      }
   }
   #pragma acc update self(image_invert[0:nx*ny]) async(1)
}

//smooth
void smooth(int* image, int* image_smooth, int nx, int ny){
   #pragma acc parallel loop copyout(image_smooth[0:nx*ny]) async(2)
   for(int j=0; j<ny; ++j){
      image_smooth[pixel(0, j, nx)]=0;
      image_smooth[pixel(nx - 1, j, nx)]=0;
   }

   #pragma acc parallel loop copyout(image_smooth[0:nx*ny]) async(2)
   for(int i=0; i<nx; ++i){
      image_smooth[pixel(i, 0, nx)] = 0;
      image_smooth[pixel(i, ny-1, nx)] = 0;
   }
   
   #pragma acc parallel loop copyin(image[0:nx*ny]) copy(image_smooth[0:nx*ny]) async(2)
   for (int j = 1; j < ny - 1; j++){
      for (int i = 1; i < nx - 1; i++){
         //image_smooth[pixel(i, j, nx)] = (image[pixel(i - 1, j - 1, nx)] * kernel[pixel(0, 0, 3)] + image[pixel(i, j - 1, nx)] * kernel[pixel(1, 0 ,3)] + image[pixel(i + 1, j - 1, nx)] * kernel[pixel(2, 0, 3)]) + (image[pixel(i - 1, j, nx)] * kernel[pixel(0, 1, 3)] + image[pixel(i, j, nx)] * kernel[pixel(1, 1 ,3)] + image[pixel(i + 1, j, nx)] * kernel[pixel(2, 1, 3)]) + (image[pixel(i - 1, j + 1, nx)] * kernel[pixel(0, 2, 3)] + image[pixel(i, j + 1, nx)] * kernel[pixel(1, 2, 3)] + image[pixel(i + 1, j + 1, nx)] * kernel[pixel(2, 2, 3)]);
         image_smooth[pixel(i, j, nx)] = image[pixel(i-1, j-1, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i-1, j, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i-1, j+1, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i, j-1, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i, j, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i, j+1, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i+1, j-1, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i+1, j, nx)];
         image_smooth[pixel(i, j, nx)] += image[pixel(i+1, j+1, nx)];
         image_smooth[pixel(i, j, nx)] = 1.0/9.0 * image_smooth[pixel(i, j, nx)];
      }
   }

   #pragma acc update self(image_smooth[0:nx*ny]) async(2)
}

//detect
void detect(int* image, int* image_detect, int nx, int ny){
   #pragma acc parallel loop copyout(image_detect[0:nx*ny]) async(3)
   for(int j=0; j<ny; ++j){
      image_detect[pixel(0, j, nx)]=0;
      image_detect[pixel(nx - 1, j, nx)]=0;
   }

   #pragma acc parallel loop copyout(image_detect[0:nx*ny]) async(3)
   for(int i=0; i<nx; ++i){
      image_detect[pixel(i, 0, nx)] = 0;
      image_detect[pixel(i, ny-1, nx)] = 0;
   }

   //float kernel[] = {0.0, -1.0, 0.0, 1.0, -4.0, 1.0, 0.0, -1.0, 0.0};
   #pragma acc parallel loop copyin(image[0:nx*ny]) copy(image_detect[0:nx*ny]) async(3)
   for (int j = 1; j < ny - 1; j++){
      for (int i = 1; i < nx - 1; i++){
         //image_detect[pixel(i, j, nx)] = (image[pixel(i - 1, j - 1, nx)] * kernel[pixel(0, 0, 3)] + image[pixel(i, j - 1, nx)] * kernel[pixel(1, 0 ,3)] + image[pixel(i + 1, j - 1, nx)] * kernel[pixel(2, 0, 3)]) + (image[pixel(i - 1, j, nx)] * kernel[pixel(0, 1, 3)] + image[pixel(i, j, nx)] * kernel[pixel(1, 1 ,3)] + image[pixel(i + 1, j, nx)] * kernel[pixel(2, 1, 3)]) + (image[pixel(i - 1, j + 1, nx)] * kernel[pixel(0, 2, 3)] + image[pixel(i, j + 1, nx)] * kernel[pixel(1, 2, 3)] + image[pixel(i + 1, j + 1, nx)] * kernel[pixel(2, 2, 3)]);
         image_detect[pixel(i, j, nx)] = image[pixel(i-1, j, nx)];
         image_detect[pixel(i, j, nx)] += image[pixel(i+1, j, nx)];
         image_detect[pixel(i, j, nx)] += image[pixel(i, j-1, nx)];
         image_detect[pixel(i, j, nx)] += image[pixel(i, j+1, nx)];
         image_detect[pixel(i, j, nx)] -= 4*image[pixel(i, j, nx)];
         
         //correct values that are out of bounds 
         image_detect[pixel(i, j, nx)]  = fmin(image_detect[pixel(i, j, nx)],255);
         image_detect[pixel(i, j, nx)]  = fmax(image_detect[pixel(i, j, nx)],0);
      }
   }

   #pragma acc update self(image_detect[0:nx*ny]) async(3)
}

//enhance
void enhance(int* image,int *image_enhance,int nx, int ny){
  #pragma acc parallel loop copyout(image_enhance[0:nx*ny]) async(4)
   for(int j=0; j<ny; ++j){
      image_enhance[pixel(0, j, nx)]=0;
      image_enhance[pixel(nx - 1, j, nx)]=0;
   }

   #pragma acc parallel loop copyout(image_enhance[0:nx*ny]) async(4)
   for(int i=0; i<nx; ++i){
      image_enhance[pixel(i, 0, nx)] = 0;
      image_enhance[pixel(i, ny-1, nx)] = 0;
   }

   //float kernel[] = {0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0};
   #pragma acc parallel loop copyin(image[0:nx*ny]) copy(image_enhance[0:nx*ny]) async(4)
   for (int j = 1; j < ny - 1; j++){
      for (int i = 1; i < nx - 1; i++){
         // image_enhance[pixel(i, j, nx)] = (image[pixel(i - 1, j - 1, nx)] * kernel[pixel(0, 0, 3)] + image[pixel(i, j - 1, nx)] * kernel[pixel(1, 0 ,3)] + image[pixel(i + 1, j - 1, nx)] * kernel[pixel(2, 0, 3)]) + (image[pixel(i - 1, j, nx)] * kernel[pixel(0, 1, 3)] + image[pixel(i, j, nx)] * kernel[pixel(1, 1 ,3)] + image[pixel(i + 1, j, nx)] * kernel[pixel(2, 1, 3)]) + (image[pixel(i - 1, j + 1, nx)] * kernel[pixel(0, 2, 3)] + image[pixel(i, j + 1, nx)] * kernel[pixel(1, 2, 3)] + image[pixel(i + 1, j + 1, nx)] * kernel[pixel(2, 2, 3)]);
         image_enhance[pixel(i, j, nx)] = -image[pixel(i-1, j, nx)];
         image_enhance[pixel(i, j, nx)] -= image[pixel(i+1, j, nx)];
         image_enhance[pixel(i, j, nx)] -= image[pixel(i, j-1, nx)];
         image_enhance[pixel(i, j, nx)] -= image[pixel(i, j+1, nx)];
         image_enhance[pixel(i, j, nx)] += 5*image[pixel(i, j, nx)]; 
         
         image_enhance[pixel(i,j,nx)]  = fmin(image_enhance[pixel(i,j,nx)],255);
         image_enhance[pixel(i,j,nx)]  = fmax(image_enhance[pixel(i,j,nx)],0);
      }
   }

   #pragma acc update self(image_enhance[0:nx*ny]) async(4)
}

 //Main program 
int main (int argc, char *argv[])
{
   int    nx,ny;
   char   filename[250];


   /* Get parameters */
   if (argc != 4) 
   {
      printf ("Usage: %s image_name N M \n", argv[0]);
      exit (1);
   }
   sprintf(filename, "%s.txt", argv[1]);
   nx  = atoi(argv[2]);
   ny  = atoi(argv[3]);

   printf("%s %d %d\n", filename, nx, ny);

   /* Allocate pointers */
   int*   image = (int *) malloc(sizeof(int)*nx*ny); 
   int*   image_invert  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_smooth  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_detect  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_enhance = (int *) malloc(sizeof(int)*nx*ny); 
   

   

   double runtime;
   /* Apply filters */
   #pragma acc enter data create(image[0:nx*ny], image_invert[0:nx*ny],image_smooth[0:nx*ny],image_detect[0:nx*ny],image_enhance[0:nx*ny])
   #pragma acc update device(image[0:nx*ny])
   double init_time = omp_get_wtime();
   invert(image, image_invert, nx, ny);
   smooth(image, image_smooth, nx, ny);
   detect(image, image_detect, nx, ny);
   enhance(image, image_enhance, nx, ny);
   //we  wait for all parallel execution finalize
   #pragma acc wait
   runtime = omp_get_wtime() - init_time;
   

   printf("Total time: %f\n",runtime);

   /* Save images */
   char fileout[255]={0};
   sprintf(fileout, "%s-inverse.txt", argv[1]);
   saveimg(fileout,nx,ny,image_invert);
   sprintf(fileout, "%s-smooth.txt", argv[1]);
   saveimg(fileout,nx,ny,image_smooth);
   sprintf(fileout, "%s-detect.txt", argv[1]);
   saveimg(fileout,nx,ny,image_detect);
   sprintf(fileout, "%s-enhance.txt", argv[1]);
   saveimg(fileout,nx,ny,image_enhance);

   #pragma acc exit data delete(image[0:nx*ny], image_invert[0:nx*ny],image_smooth[0:nx*ny],image_detect[0:nx*ny],image_enhance[0:nx*ny])
   /* Deallocate  */
   free(image);
   free(image_invert);
   free(image_smooth);
   free(image_detect);
   free(image_enhance);

}
