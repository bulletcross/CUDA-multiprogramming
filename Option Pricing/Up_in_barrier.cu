#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Normal random generation using Marsaglia method
//adding __host__ __device__ creates two version of function, one callable from cpu, other from gpu
__host__ __device__ float normal_random (float mu, float sigma)
{
  float U1,U2,W;
  float normal_random;
  do{
      U1 = -1 + ((float) rand () / RAND_MAX) * 2;
      U2 = -1 + ((float) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
  }while (W >= 1 || W == 0);
  normal_random = U1 * sqrt ((-2 * log (W)) / W);
  printf("Function called");
  return (mu + sigma * (float)normal_random);
}

//Kernel function for genrating a random path
__global__ void random_path(float *d_prices,float *d_output,float S_0,float T,float rate,float volatility,float strike,float barrier,int N_THREAD,int N_STEPS){
	//Generate a random path by generating normal random number
	//Maximum path length is N_STEPS
	//Compute the expected option price
	//Reduce to combine all of the simulation

	int t_id = blockDim.x*blockIdx.x + threadIdx.x;
	
	if(t_id>N_THREAD){
		return;
	}

	int other_id;
	float delta_t = (float)N_STEPS/T;
	int i;
	float normal,option_price;
	int in=0;
	for(i=0;i<N_STEPS;i++){
		normal = normal_random(0,sqrt(delta_t));
		S_0 = S_0 + rate*S_0*delta_t + volatility*S_0*normal;
		if(in==0 && S_0>barrier){
			in = 1;
		}
	}
	option_price = (float) exp(-rate*T)*in*(S_0-strike);
	d_prices[t_id] = option_price;
	__syncthreads();

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
    }
}

//*******************************************Main Function*********************************//
int main(){
	//Variable declaration
	float S_0,T;
	float strike,barrier;
	float rate,volatility;
	float *h_output;
	clock_t startTime;

	int N_THREAD;
	int N_STEPS;

	//Setting variables
	S_0 = 200,T=365,strike = 190,barrier = 210,rate = 0.5, volatility = 1;
	N_THREAD = 5000;
	N_STEPS = 365;

	//Variable for device
	float *d_prices;
	float *d_output;

	//Seeding the random generator
	int seed = time(NULL);
    srand(seed);

    //Memory allocation
    h_output = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&d_prices,sizeof(float)*N_THREAD);
	cudaMalloc((void**)&d_output,sizeof(float));

	//TO-DO Take the N_THREAD from system
	dim3 dimBlock(1024,1,1);
	dim3 dimGrid((int)(N_THREAD+1023)/1024,1,1);
	startTime = clock();
	//Calling Kernal funtion to execute in on device
	random_path<<<dimGrid,dimBlock>>>(d_prices,d_output,S_0,T,rate,volatility,strike,barrier,N_THREAD,N_STEPS);
	cudaMemcpy((void *)h_output,(void *)d_output,sizeof(float),cudaMemcpyDeviceToHost);
	printf("%f",h_output[0]);
	printf("Time for parallel DFA with O(log(n)) reduction is %lf\n",(double)(( clock() - startTime ) / (double)CLOCKS_PER_SEC));
	
	return 0;
}