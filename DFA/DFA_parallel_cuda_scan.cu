
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//Kernel function - Given transition matrix and a part of input string,  thread creates an array of possible states
//Reduction of state vector is done with O(log(P)) - P is number of threads in execution
__global__  void DFA_kernal(int *t_m,int *in,int n_state,int n_sigma,int init_state,int final_state,int n,int *out,int m,int n_t) {

	//Shared arrays of state vector for faster retrival and writing, it is vector of state vectors for all threads.
	extern __shared__ int state_vectors[];

	//Variable used
	int i,j,k;
	//Thread ID assuming 1D block structure in 1D Grid structure
	int t_id = threadIdx.x;
	int other_t_id;
	//Initial retrival of possible state corresponding to first input
	for(i=0;i<n_state;i++){
		state_vectors[n_state*t_id + i] = t_m[n_sigma*i + in[t_id*m]];
	}
	// This thread will act on [i,j] part on input of length n
	i=t_id*m+1;
	j=(t_id+1)*m -1;
	//Checks for boundary condition
	if(j>=n){
		j=n-1;
	}

	//Reuse the state vector alloted to the thread to compute final state vector
	while(i<=j){
		for(k=0;k<n_state;k++){
			state_vectors[n_state*t_id + k] = t_m[n_sigma*state_vectors[n_state*t_id + k] + in[i]];
		}
		i++;
	}
	//Wait till all vector finishes
	__syncthreads();

	//O(log(n_t)) reduction. n_t is total number of threads in use in the given block

    for(i=1;i<=(int)log2((float)n_t);i++){
    	if((t_id+1)%(int)pow((double)2,(double)i) == 0){
    		other_t_id = t_id - (int)pow((double)2,(double)i-1);
    		for(k=0;k<n_state;k++){
    			state_vectors[n_state*t_id + k] = state_vectors[n_state*t_id + state_vectors[n_state*other_t_id + k]];
    		}
    	}
    	__syncthreads();
    }

    //Final output is copied to global device memory
    if(t_id==0){
    	for(i=0;i<n_state;i++){
    		out[i] = state_vectors[n_state*(n_t-1) + i];
    	}
    }
}
 
 /*******************************************Main Function starts here*******************************************/

int main()
{
	//Variables requred for transition function and input memory allocation
	int STATES,SIGMA,INITIAL_STATE,FINAL_STATE,INPUT_LENGTH;
	int i,j;

	//Taking input
	scanf("%d %d %d %d",&STATES,&SIGMA,&FINAL_STATE,&INPUT_LENGTH);
	
	//Number of threads depend on optimality condition
	int NUM_THREADS=4;
	int ELEM_PER_THREAD= ((INPUT_LENGTH%NUM_THREADS==0) ? INPUT_LENGTH/NUM_THREADS:INPUT_LENGTH/NUM_THREADS + 1);
	//An additional state has to be added for complete transition function
	STATES++;
	INITIAL_STATE = 0;
	
	//Input memory allocation and input retrival
	int *input = (int *)malloc(sizeof(int)*INPUT_LENGTH);
	for(i=0;i<INPUT_LENGTH;i++){
		scanf("%d",&input[i]);
	}
	
	//Allocating memory and retriving to transition matrix
	int **transition_matrix = (int **)malloc(sizeof(int *)*STATES);
	int *transition_matrix_data = (int *)malloc(sizeof(int)*STATES*SIGMA);
	for(i=0;i<STATES;i++){
		transition_matrix[i] = &transition_matrix_data[i*SIGMA];
	}
	for(i=0;i<STATES;i++){
		for(j=0;j<SIGMA;j++){
			scanf("%d",&transition_matrix[i][j]);
		}
	}
	
	//printing the input taken
	for(i=0;i<INPUT_LENGTH;i++){
		printf("%d ",input[i]);
	}
	printf("\n");
	for(i=0;i<STATES;i++){
		for(j=0;j<SIGMA;j++){
			printf("%d ",transition_matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	//final output memory requred by device 
	int *h_out = (int *)malloc(sizeof(int)*STATES);

 	//Device memory
	int *d_transition_matrix;
	int *d_input;
	int *d_output;
 	
 	//Allocating and initializing memory on GPU
	cudaMalloc((void**)&d_transition_matrix,sizeof(int)*STATES*SIGMA);
	cudaMemcpy((void *)d_transition_matrix,(void *)transition_matrix_data,sizeof(int)*STATES*SIGMA,cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&d_input,sizeof(int)*INPUT_LENGTH);
	cudaMemcpy((void *)d_input,(void *)input,sizeof(int)*INPUT_LENGTH,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_output,sizeof(int)*STATES);
	
	//Declaring grid and block size
	dim3 dimBlock(NUM_THREADS,1,1);
	dim3 dimGrid(1,1,1);

	//Calling Kernal funtion to execute in on device
	DFA_kernal<<<dimGrid, dimBlock, STATES*NUM_THREADS>>>(d_transition_matrix,d_input,STATES,SIGMA,INITIAL_STATE,FINAL_STATE,INPUT_LENGTH,d_output,ELEM_PER_THREAD,NUM_THREADS);

	//Copy back the computed result from device to host
	cudaMemcpy((void *)h_out,(void *)d_output,sizeof(int)*STATES,cudaMemcpyDeviceToHost);

	//Freeing-up the GPU intialized memory
	cudaFree(d_output);
	cudaFree(d_transition_matrix);
	cudaFree(d_input);

	//Print the compueted output
	for(i=0;i<STATES;i++){
		printf("%d\n",h_out[i]);
	}

	//Condition checking
	if(h_out[0]==FINAL_STATE){
		printf("Automata is accepting the string\n");
	}
	else{
		printf("String not accepted\n");
	}

	printf("All done\n");

	return 0;
}

/***********************************************Main ends here**********************************************************/