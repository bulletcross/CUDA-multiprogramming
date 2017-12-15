__kernel void trans_level2(__global const int* A, __global int* B, const int M, const int N){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);
  __local int temp[32][32];
  temp[y_local][x_local] = A[x*N + y];
  barrier(CLK_LOCAL_MEM_FENCE);
  x = get_group_id(1)*32 + x_local;
  y = get_group_id(0)*32 + y_local;
  B[x*M + y] = temp[x_local][y_local];
}
