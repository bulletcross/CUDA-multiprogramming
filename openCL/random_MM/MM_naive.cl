__kernel void MM_naive(__global const int* A, __global const int* B, __global int *C){
  int x = get_global_id(0);
  int y = get_global_id(1);
  C[x*1024 + y] = 0;
  for(int i=0;i<1024;i++){
    C[x*1024 + y] += A[x*1024 + i]*B[i*1024 + y];
  }
}
