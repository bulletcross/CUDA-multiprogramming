__kernel void trans_level1(__global const int* A, __global int* B, const int N, const int M){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);
  int x_group = get_group_id(0);
  int y_group = get_group_id(1);
  __local int temp[16][32];
  temp[y_local][x_local] = A[x*M + y];
  barrier(CLK_LOCAL_MEM_FENCE);
  B[(y_group*16 + y_local)*N + x_group*32 + x] = temp[y_local][x_local];
}
