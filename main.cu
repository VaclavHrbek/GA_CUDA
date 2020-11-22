#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include <math.h>
#include<curand.h>


#define SIZE_POP 5000
//#define SIZE_IND 40	//Fixed length
//#define SIZE_PL 10
#define N_GEN 200

#define P_CROSS 48
#define P_MUT 48

#define UPPER 254
#define LOWER 0

//Load data define thing function
#define FILE_NAME "DataValues.csv"
#define NUM_ROWS 40
#define NUM_COLUMS 2
#define COMA " ,"


int8_t randomRangeInt(int lower, int upper){
    return ((int) round((rand() % (upper -  lower +1))+lower));
}
double randomRange(int lower, int upper){
    return ((double) (rand() % (upper -  lower +1))+lower);
}
void createInitRandPop(int8_t *pop){
	for(int i = 0; i < SIZE_POP*4; i++){
		*(pop + i) = round(randomRange(0, 254));
	}
}
void printPopulation(int8_t *pop){
	for(int i = 0; i < SIZE_POP;i++){
		for(int j = 0; j < 4; j++){
			printf("%d ", *(pop + i*4 + j));
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
//Insertion sort algorithm
void insertionSort(double *fit, int8_t *pop){
    int arr_lenght = SIZE_POP;
    for(int k = 0; k < arr_lenght - 1; k++){
        for(int i = k + 1; i < arr_lenght; i++){
            if(*(fit + k) > *(fit + i)){
                //helper variables
                int8_t p_hlp[4] = {0};
                double f_hlp = 0;

                //switch fitness
                f_hlp = *(fit + k);
                *(fit + k) = *(fit + i);
                *(fit + i) = f_hlp;

                //switch individuals
                //Copy array[i] to hlp[]
                for(int j = 0; j < 4; j++){
                    p_hlp[j] = *(pop + k*4 + j);
                }
                for(int j = 0; j < 4; j++){
                    *(pop + k*4 + j) = *(pop + i*4 + j);
                }
                for(int j = 0; j < 4; j++){
                    *(pop + i*4 + j) = p_hlp[j];
                }
            }

        }

    }
}
int *indDecToBin(int8_t a,int8_t b,int8_t c,int8_t d){
    int *bin = (int *)malloc(sizeof(int)* 32);
    int i = 0;
    //printf("%d\n",a);
    int aa[8];
    for(int j = 0; j < 8; j++){
        if( a > 0){
            aa[j] = a % 2;
            a = a / 2;
        }
        else{
            aa[j] = 0;
        }
        //printf("%d",aa[j]);
    }
    //printf("\n");
    for(int j = 7; j >= 0; j--){
    //    printf("%d",aa[j]);
        bin[i] = aa[j];
        i++;
    }

//    printf("%d\n",b);
    int ab[8];
    for(int j = 0; j < 8; j++){
        if( b > 0){
            ab[j] = b % 2;
            b = b / 2;
        }
        else{
            ab[j] = 0;
        }
    //    printf("%d",ab[j]);
    }
    //printf("\n");
    for(int j = 7; j >= 0; j--){
    //    printf("%d",ab[j]);
        bin[i] = ab[j];
        i++;
    }

    //printf("%d\n",c);
    int ac[8];
    for(int j = 0; j < 8; j++){
        if( c > 0){
            ac[j] = c % 2;
            c = c / 2;
        }
        else{
            ac[j] = 0;
        }
    //    printf("%d",ac[j]);
    }
    //printf("\n");
    for(int j = 7; j >= 0; j--){
    //    printf("%d",ac[j]);
        bin[i] = ac[j];
        i++;
    }

    //printf("%d\n",d);
    int ad[8];
    for(int j = 0; j < 8; j++){
        if( d > 0){
            ad[j] = d % 2;
            d = d / 2;
        }
        else{
            ad[j] = 0;
        }
    //    printf("%d",ad[j]);
    }
    //printf("\n");
    for(int j = 7; j >= 0; j--){
    //    printf("%d",ad[j]);
        bin[i] = ad[j];
        i++;
    }
    return bin;
}
int8_t get_a(int *ind){
    int8_t a = 0;
    for(int i = 0; i < 8; i++){
        a += ind[i]*pow(2,8-i);

    }

    return a;
}
int8_t get_b(int *ind){
    int8_t x = 0;
    for(int i = 0; i < 8; i++){
        x += ind[i+8]*pow(2,8-(i+1));
        //printf("%d\n",ind[i+8]);
    }
    return x;
}
int8_t get_c(int *ind){
    int8_t x = 0;
    for(int i = 0; i < 8; i++){
        x += ind[i+16]*pow(2,8-(i+1));
        //printf("%d\n",ind[i+16]);
    }
    return x;
}
int8_t get_d(int *ind){
    int8_t x = 0;
    for(int i = 0; i < 8; i++){
        x += ind[i+24]*pow(2,8-(i+1));
        //printf("%d\n",ind[i+24]);
    }
    return x;
}

void geneticOperations(int8_t *pop, int8_t *new_pop){
    //Mutation
    int n_best_m = round(((double)SIZE_POP / 100) * P_MUT);
    int i;
    for(i = 0; i < n_best_m; i++){
        //random number between 0 - 4
        int rand = round(randomRange(0, 32));
        for(int j = 0; j < 4; j++){
            *(new_pop + i*4 + j) = *(pop +i*4 + j);
        }

        int *ind = indDecToBin(*(new_pop +i*4 +0),*(new_pop +i*4 +1),
            *(new_pop +i*4 +2),*(new_pop +i*4 +3));
        ;
            //printf("\n");
        /*for(int j = 0; j < 32; j++){
            printf("%d", ind[j]);
        }*/

        if(ind[rand] == 1){
            ind[rand] = 0;
        }
        else{
            ind[rand] = 1;
        }

        /*for(int j = 0; j < 32; j++){
            printf("%d", ind[j]);
        }*/


         *(new_pop +i*4 +0) = get_a(ind);
    //    printf("### a = %d\n", a);
         *(new_pop +i*4 +1) = get_b(ind);
    //    printf("### d = %d\n", b);
         *(new_pop +i*4 +2) = get_c(ind);
    //    printf("### c = %d\n", c);
         *(new_pop +i*4 +3) = get_d(ind);
    //    printf("### d = %d\n", d);
    }

    //crossover
    int n_best_c = round(((double)SIZE_POP / 100) * P_CROSS) + n_best_m;
    int x = 0;
    /*printf("i = %d | n_best_m = %d | n_best_c = %d\n", i,n_best_m,n_best_c-n_best_m);
    printf("pop:\n");
    printPopulation(new_pop);*/
    for(i ; i < n_best_c; i = i + 2){
            if((x+1) >= SIZE_POP){
                x = 0;
            }
            int *ind1 = indDecToBin(*(pop +x*4 +0),*(pop +x*4 +1),
                *(pop +x*4 +2),*(pop +x*4 +3));
            int *ind2 = indDecToBin(*(pop +(x+1)*4 +0),*(pop +(x+1)*4 +1),
                *(pop +(x+1)*4 +2),*(pop +(x+1)*4 +3));

            int new_ind1[32] = {0};
            int new_ind2[32] = {0};

            int rand = round(randomRange(0, 32));
            for(int k = 0; k < 32; k++){
                if(k<rand){
                    new_ind1[k] = ind1[k];
                    new_ind2[k] = ind2[k];
                }
                else{
                    new_ind1[k] = ind2[k];
                    new_ind2[k] = ind1[k];
                }

            }
            x = x + 2;
            /*for(int b = 0; b < 32; b++){
                printf("%d",ind1[b]);
            }
            printf("\n");
            for(int b = 0; b < 32; b++){
                printf("%d",ind2[b]);
            }
            printf("\n");
            for(int b = 0; b < 32; b++){
                printf("%d",new_ind1[b]);
            }
            printf("\n");
            for(int b = 0; b < 32; b++){
                printf("%d",new_ind2[b]);
            }
            printf("\n");*/
            *(new_pop +i*4 +0) = get_a(new_ind1);
       //    printf("### a = %d\n", a);
            *(new_pop +i*4 +1) = get_b(new_ind1);
       //    printf("### d = %d\n", b);
            *(new_pop +i*4 +2) = get_c(new_ind1);
       //    printf("### c = %d\n", c);
            *(new_pop +i*4 +3) = get_d(new_ind1);
       //    printf("### d = %d\n", d);
           *(new_pop +(i+1)*4 +0) = get_a(new_ind2);
      //    printf("### a = %d\n", a);
           *(new_pop +(i+1)*4 +1) = get_b(new_ind2);
      //    printf("### d = %d\n", b);
           *(new_pop +(i+1)*4 +2) = get_c(new_ind2);
      //    printf("### c = %d\n", c);
           *(new_pop +(i+1)*4 +3) = get_d(new_ind2);
      //    printf("### d = %d\n", d);
     /* printf("i = %d | n_best_m = %d | n_best_c = %d | k = %d\n", i,n_best_m,n_best_c-n_best_m, rand);
      printf("pop:\n");
      printPopulation(new_pop);*/
    }

    //Reproduction
    int j = 0;
    for(i; i < SIZE_POP; i++){
        for(int k = 0; k < 4; k++){
                    *(new_pop + i*4 + k) = *(pop +j*4 + k);
        }
        j++;
    }
}

__global__ void calculateFitness(const int8_t *pop, double *fit, const double *data){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < SIZE_POP){
        double arr_polyn[NUM_ROWS];

            for(int j = 0; j < NUM_ROWS; j++){
                double out_polyn = 0;
                for (int k = 0; k < 4; k++){
                    out_polyn += (*(pop + i*4 + k)*(pow(*(data + j*2 + 0), ((double)(4-(k+1))))));
                }
                 arr_polyn[j] = out_polyn;
	        }
            double fitness_function = 0;
            for(int j = 0; j < NUM_ROWS; j++){
                fitness_function += pow((*(data + j*2 + 1) - arr_polyn[j]), 2);
            }
            *(fit + i) = fitness_function;
        }
}

int main(void){

    //Inicialize random number generaion
    time_t t;
    srand((unsigned) time(&t));
	//Load data from .csv file
	double *data = (double *)malloc(NUM_ROWS*NUM_COLUMS*sizeof(double));
	loadData(data);
	//error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
	//variable for size of population;
	size_t sizePop = 4 * SIZE_POP * sizeof(int8_t);
    size_t sizeFit = SIZE_POP * sizeof(double);

	//allocation of memory for population on host
	int8_t *population = (int8_t *)malloc(sizePop);
    int8_t *new_population = (int8_t *)malloc(sizePop);
    double *fitness = (double *)malloc(sizeFit);
	//verification of succesed allocation
	if(population == NULL || data == NULL || fitness == NULL || new_population == NULL){
		fprintf(stderr, "Failed to allocate host memory\n");
		exit(EXIT_FAILURE);
	}

	//allocating of memory for population on device
	int8_t *d_population = NULL;
	err = cudaMalloc((void **)&d_population, sizePop);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed allocate memory for population on device(error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	//allocating of memory for data on device
	double *d_data = NULL;
	err = cudaMalloc((void **)&d_data, (NUM_ROWS*NUM_COLUMS*sizeof(double)));
	//checking succeded allocation on device memory
	if(err != cudaSuccess){
		fprintf(stderr, "Failed allocate memory for data on device(error code %s)\n",cudaGetErrorString(err));
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

    	//BEGINING OF GA
        int e = 0;
    do{
        e++;
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
            err = cudaMemcpy(fitness, d_fitness, sizeFit, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy fitness from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
    		cudaDeviceSynchronize();
            //sort population
            insertionSort(fitness, population);

            /*printPopulation(population);
            for(int c = 0; c < SIZE_POP; c++){
                printf("%f\n", fitness[c]);
            }*/
            printf("%d, %d, %d, %d | fitness = %f || gen: %d\n", *(population + 0),*(population + 1),
            *(population + 2),*(population + 3),*(fitness + 0), gen);
            //apply genetioc operation
            geneticOperations(population, new_population);
            //copy new population to old population
            for(int i = 0; i < SIZE_POP; i++){
                for(int j = 0; j < 4; j++){
                    *(population + i*4 +j) = *(new_population + i*4 + j);
                }
            }

    		//--------
    		gen++;
    	}
    }while(*(fitness) != 0);
    free(population);
    free(new_population);
    free(data);
    free(fitness);
    cudaFree(d_population);
    cudaFree(d_data);
    cudaFree(d_fitness);


	return 0;
}
