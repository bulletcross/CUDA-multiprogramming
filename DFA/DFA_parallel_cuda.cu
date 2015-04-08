#include <stdio.h>
#include <stdlib.h>

int *final_state;

//Kernel
__global__  void DFA_kernal(int *t_m,int *in,int n_state,int n_sigma,int init_state,int final_state,int n) {
	int i;
	int t_id = threadIdx.x;
	int *state_vector = (int *)malloc(sizeof(int)*n_state);
	//TO-DO: give this thread a part of string
	for(i=0;i<n_state;i++){
		state_vector[i] = t_m[n_sigma*i + in[t_id]];
	}
	__syncthreads();

}
 
int main()
{
	//Variables
	int STATES,SIGMA,INITIAL_STATE,FINAL_STATE,INPUT_LENGTH;
	int i,j;
	//Taking input
	//cin >> STATES >> SIGMA >> FINAL_STATE >> INPUT_LENGTH;
	scanf("%d %d %d %d",&STATES,&SIGMA,&FINAL_STATE,&INPUT_LENGTH);
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
	//////////////////////////////////////////////////////////////////////////
 	//Device memory
	int *d_transition_matrix;
	int *d_input;
 	
 	//Allocating and initializing memory on GPU
	cudaMalloc((void**)&d_transition_matrix,sizeof(int)*STATES*SIGMA);
	cudaMemcpy((void *)d_transition_matrix,(void *)transition_matrix_data,sizeof(int)*STATES*SIGMA,cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&d_input,sizeof(int)*INPUT_LENGTH);
	cudaMemcpy((void *)d_input,(void *)input,sizeof(int)*INPUT_LENGTH,cudaMemcpyHostToDevice);
	
	//Declaring grid and block size
	dim3 dimBlock(INPUT_LENGTH,1,1);
	dim3 dimGrid(1,1,1);
	DFA_kernal<<<dimGrid, dimBlock>>>(d_transition_matrix,d_input,STATES,SIGMA,INITIAL_STATE,FINAL_STATE,INPUT_LENGTH);

	cudaFree(d_transition_matrix);
	cudaFree(d_input);

	printf("All done\n");
	return 0;
}