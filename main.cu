#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include <math.h>


#define SIZE_POP 1000
#define SIZE_IND 40	//Fixed length
#define SIZE_PL 10
#define N_GEN 1

//Load data define thing function
#define FILE_NAME "DataValues.csv"
#define NUM_ROWS 40
#define NUM_COLUMS 2
#define COMA " ,"

int randomRangeInt(int lower, int upper){
    return ((int) round((rand() % (upper -  lower +1))+lower));
}
double randomRange(int lower, int upper){
    return ((double) (rand() % (upper -  lower +1))+lower);
}
void createInitRandPop(int *pop){
	for(int i = 0; i < (SIZE_POP*SIZE_IND); i++){
		*(pop + i) = randomRangeInt(0, 1);
	}
}
void printPopulation(int *pop){
	for(int i = 0; i < SIZE_POP;i++){
		int k = 0;
		for(int j = 0; j < SIZE_IND; j++){
			printf("%d", *(pop + i*SIZE_POP + j));
			if(k == 9){
				k = -1;
				printf(" ");
			}
			k++;

		}
		printf("\n");
	}
}
void loadData(double *data){

  //double data[NUM_ROWS][NUM_COLUMS];
  FILE* data_file = fopen(FILE_NAME, "r");
  char line[NUM_ROWS];
  int i = 0;

  while(fgets(line, sizeof(line), data_file)){
    char* tok = strtok(line, COMA);
    int j = 0;

    while(tok != NULL){
        *(data + i*NUM_COLUMS + j) = atof(tok);   //const char to double
        tok = strtok(NULL, COMA);
        j++;
      }
      i++;



  }
}

__global__ void calculateFitness(int *pop, double *fit, double *data){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < SIZE_POP){
		int q = 9;
		int a = 0;

		for(int j = 0; j < SIZE_PL; j++){
			a = a + pow((*(pop + i*SIZE_IND + j))*2, q);
						printf("%d",*(pop + i*SIZE_IND + j));
			q = q - 1;

		}
		printf("\n%d\n",a);

	}



}

int main(void){

	//Load data from .csv file
	double *data = (double *)malloc(NUM_ROWS*NUM_COLUMS*sizeof(double));
	loadData(data);
	//Inicialize random number generaion
	time_t t;
	srand((unsigned) time(&t));
	//error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	//variable for size of d_population;
	size_t sizePop = SIZE_POP * SIZE_IND *sizeof(int);
	size_t sizeFit = SIZE_POP * sizeof(double);

	//allocation of memory for population on host
	int *population = (int *)malloc(sizePop);
	//allocating memory for fitness fuction on host
	double *fitness = (double *)malloc(sizeFit);
	//verification of succesed allocati
	if(population == NULL || fitness == NULL || data == NULL){
		fprintf(stderr, "Failed to allocate host memory\n");
		exit(EXIT_FAILURE);
	}

	//allocating of memory for population on device
	int *d_population = NULL;
	err = cudaMalloc((void **)&d_population, sizePop);
	//checking succeded allocation on device memory
	if(err != cudaSuccess){
		fprintf(stderr, "Failed allocate memory for population on device(error code %s)\n",cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//allocating of memory for fitness on device
	double *d_fitness = NULL;
	err = cudaMalloc((void **)&d_fitness, sizeFit);
	//checking succeded allocation on device memory
	if(err != cudaSuccess){
		fprintf(stderr, "Failed allocate memory for fitness on device(error code %s)\n",cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//allocating of memory for fitness on device
	double *d_data = NULL;
	err = cudaMalloc((void **)&d_data, (NUM_ROWS*NUM_COLUMS*sizeof(double)));
	//checking succeded allocation on device memory
	if(err != cudaSuccess){
		fprintf(stderr, "Failed allocate memory for data on device(error code %s)\n",cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//BEGINING OF GA
	//--------
	//Creating initial population
	createInitRandPop(population);
	//--------
	//Checking termination criteria
	int gen = 0;
	while(gen != N_GEN){
		//--------
		//Evaluate fitness
		//copy population and data from host to device
		err = cudaMemcpy(d_population, population, sizePop, cudaMemcpyHostToDevice);
		if(err != cudaSuccess){
			fprintf(stderr, "Failed copy population to device errCode: %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(d_data, data, (NUM_ROWS*NUM_COLUMS*sizeof(double)), cudaMemcpyHostToDevice);
		if(err != cudaSuccess){
			fprintf(stderr, "Failed copy data to device errCode: %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		int threadsPerBlock = 256;
		int blocksPerGrid = (SIZE_POP + threadsPerBlock - 1) / threadsPerBlock;
		//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		calculateFitness<<<blocksPerGrid, threadsPerBlock>>>(d_population, d_fitness, d_data);
		cudaDeviceSynchronize();

		printf("\n");
		printPopulation(population);
		//--------
		gen++;
	}


	return 0;
}
