__kernel void MM_level2(__global const int* A, __global const int* B, __global int *C){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);
  int ret = 0;
  __local int A_tile_temp[16][16];
  __local int B_tile_temp[16][16];

  for(int tile = 0; tile<64; tile++){
    A_tile_temp[x_local][y_local] = A[x*1024 + (tile*16 + y_local)];
    B_tile_temp[x_local][y_local] = B[(tile*16 + x_local)*1024 + y];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k=0;k<16;k++){
      ret+= A_tile_temp[x_local][k]*B_tile_temp[k][y_local];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[x*1024 + y] = ret;
}
