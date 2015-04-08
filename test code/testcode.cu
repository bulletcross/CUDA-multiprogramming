#include <stdio.h>
#include <stdlib.h>

const int N = 16; 
const int blocksize = 20; 

//Kernel
__global__  void block(int *d_ar,int size) {
	//Local memory
	int t_id = threadIdx.x;
	if(t_id>=size){
		return;
	}
	printf("Thread %d got %d\n",t_id,d_ar[t_id]);
}

void create_data(int *h_ar){
	for(int i=0;i<N;i++){
		h_ar[i] = (rand()%10) + 1;
	}
}
 
int main()
{
	int size = N*sizeof(int);
	//Host memory
	int *h_ar  = (int *)malloc(size);
 	create_data(h_ar);
 	//Device memory
	int *d_ar;
 	
 	//Allocating and initializing memory on GPU
	cudaMalloc( (void**)&d_ar,size);
	cudaMemcpy( d_ar, h_ar, size, cudaMemcpyHostToDevice );
	
	//Declaring grid and block size
	dim3 dimBlock( blocksize, 1 ,1);
	dim3 dimGrid( 1, 1 ,1);
	block<<<dimGrid, dimBlock>>>(d_ar,N);

	cudaFree( d_ar );
	
	printf("All done\n");
	return 0;
}