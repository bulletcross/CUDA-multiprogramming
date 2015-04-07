__global__ void shared_reduce(float *d_out, const float *d_in){
	external __shared__ float sdata[];
	int myId = threadIdx.x + blockDim.x*blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = d_in[myid];
	__syncthreads();

	for(unsigned int s = blockDim.x/2; s>0;s>>=1){
		if(tid<s){
			sdata[tid]+=sdata[tid+s];
		}
		__syncthreads();
	}
	if(tid == 0){
		d_out[blockIdx.x] = sdata[0];
	}
}

//Calling the kernal with specified amount of external shared memory
shared_reduced<<<blocks,threads,threads*sizeof(float)>>>(d_out,d_in);