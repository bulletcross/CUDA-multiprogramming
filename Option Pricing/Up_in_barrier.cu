#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>//Generating random number on CUDA
#include <curand_kernel.h>

//Initializes the seed on each thread
__global__ void setup_kernel ( curandState * state, unsigned long seed ){
    int t_id = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init ( seed, t_id, 0, &state[t_id] );
}

//adding __host__ __device__ creates two version of function, one callable from cpu, other from gpu
__host__ __device__ float maximum(float a,float b){
	if(a<b){
		return b;
	}
	else{
		return a;
	}
}

//Normal random generation using Marsaglia method
//adding __host__ __device__ creates two version of function, one callable from cpu, other from gpu
float normal_random (float mu, float sigma){
  float U1,U2,W;
  float normal_random;
  do{
      U1 = -1 + ((float) rand () / RAND_MAX) * 2;
      U2 = -1 + ((float) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
  }while (W >= 1 || W == 0);
  normal_random = U1 * sqrt ((-2 * log (W)) / W);
  return (mu + sigma * (float)normal_random);
}

//Kernel function for genrating a random path
__global__ void random_path(float *d_prices,float *d_output,curandState* globalState,float S_0,float T,float rate,float volatility,float strike,float barrier,int N_THREAD,int N_STEPS){
	//Generate a random path by generating normal random number
	//Maximum path length is N_STEPS
	//Compute the expected option price
	//Reduce to combine all of the simulation

	int t_id = blockDim.x*blockIdx.x + threadIdx.x;
	curandState localState = globalState[t_id];
	if(t_id>N_THREAD){
		return;
	}

	int other_id;
	float delta_t = (float)N_STEPS/T;
	int i;
	float normal,option_price;
	float temp;
	int in=0;
	for(i=0;i<N_STEPS;i++){
		normal = curand_normal(&localState);
		//globalState[t_id] = localState;
		//printf("%f\n",normal);
		//delta_t is one
		S_0 = S_0 + rate*S_0*delta_t + volatility*S_0*normal;
		if(in==0 && S_0>barrier){
			in = 1;
		}
	}
	option_price = (float) exp(-rate*T)*in*maximum(0,(S_0-strike));
	//printf("%f\n",option_price);
	d_prices[t_id] = (float)option_price/N_THREAD;
	__syncthreads();

	if(t_id==0){
		temp = d_prices[0];
	for(i=1;i<N_THREAD;i++){
		temp = temp + d_prices[i];
	}
	printf("Parallel output %f\n",temp);
	d_output[0] = temp;
	}
	__syncthreads();
	
	/*
	//Reduction on GPU global memory in O(log(n))
	for(i=1;i<=(int)log2((float)N_THREAD);i++){
    	if((t_id+1)%(int)pow((double)2,(double)i) == 0){
    		other_id = t_id - (int)pow((double)2,(double)i-1);
    		d_prices[t_id] = d_prices[t_id] + d_prices[other_id];
    	}
    	__syncthreads();
    }
    if(t_id==0){
    	d_output[0] = d_prices[N_THREAD-1];
    	printf("Parallel %f\n",d_prices[N_THREAD-1]);
    }
	*/
}

//*******************************************Main Function*********************************//
int main(){
	//Variable declaration
	float S_0 = 200,T=120;
	float strike=190,barrier=210;
	float rate=0.5,volatility=1;
	float *h_output;
	clock_t startTime;

	//Variables to generate normal random number on cuda
	//curandGenerator_t curandGenerator;
	
	//Variable for device
	float *d_prices;
	float *d_output;
	float normal=0;
	cudaDeviceProp property;
	cudaGetDeviceProperties(&property, 0);
	int grid = 256;
	int N_THREAD=property.maxThreadsPerBlock*grid;
	int N_STEPS=120;
	/*float *h_normal = (float *)malloc(sizeof(float)*N_THREAD*N_STEPS);
	float *d_normal;*/

	curandState* devStates;
    cudaMalloc(&devStates,N_THREAD*N_STEPS*sizeof(curandState));

	//Seeding the random generator
	int seed = time(NULL);
    srand(seed);

    float host_option_price=0;
    float temp;
    int i,j,in;

    startTime = clock();
    for(i=0;i<N_THREAD;i++){
    	temp = 0;
    	in=0;
    	temp = S_0;
    	for(j=0;j<N_STEPS;j++){
			normal = normal_random (0,1);
			temp = temp + rate*temp + volatility*temp*normal;
			if(in==0 && temp>barrier){
				in = 1;
			}
    	}
    	temp = (float) exp(-rate*T)*in*maximum(0,(temp-strike));
		//printf("%f\n",option_price);
		temp = (float)temp/N_THREAD;
		host_option_price = host_option_price + temp;
    }
	printf("%f\t",host_option_price);
	printf("Linear time on cpu is %lf\n",(double)(( clock() - startTime ) / (double)CLOCKS_PER_SEC));

	/*cudaMalloc((void **)&d_normal,sizeof(float)*N_THREAD*N_STEPS);
	
	//Setting up generator for normal generation
	//Based on mersener twister
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
	//Setting seed
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);
	//Generating random number
	curandGenerateNormal(curandGenerator, h_normal, N_THREAD*N_STEPS, 0, 1);

	cudaMemcpy((void *)d_normal,(void *)h_normal,sizeof(float),cudaMemcpyHostToDevice);
	*/
	
    //Memory allocation
    h_output = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&d_prices,sizeof(float)*N_THREAD);
	cudaMalloc((void**)&d_output,sizeof(float));

	printf("%d  %d  %d \n",property.maxThreadsPerBlock,property.maxThreadsDim[0],property.maxGridSize[0]);
	dim3 dimBlock(property.maxThreadsPerBlock,1,1);
	dim3 dimGrid(grid,1,1);
	startTime = clock();
	//Calling Kernal funtion to execute in on device
	setup_kernel<<<dimGrid,dimBlock>>>(devStates,time(NULL));
	random_path<<<dimGrid,dimBlock>>>(d_prices,d_output,devStates,S_0,T,rate,volatility,strike,barrier,N_THREAD,N_STEPS);
	cudaMemcpy((void *)h_output,(void *)d_output,sizeof(float),cudaMemcpyDeviceToHost);
	printf("%f\t",h_output[0]);
	printf("Time for parallel is %lf\n",(double)(( clock() - startTime ) / (double)CLOCKS_PER_SEC));
	
	return 0;
}