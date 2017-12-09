__kernel void MM_naive(__global int* A, __global int* B, __global int *C){
  int x = get_group_id(0);
  int y = get_group_id(1);
  for(int i=0;i<1024;i++){
    C[x*1024 + y] += A[x*1024 + i]*B[i*1024 + y];
  }
}
