__kernel void MM_level3(__global const int* A, __global const int* B, __global int *C){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);
  __local int A_tile_temp[16][16];
  __local int B_tile_temp[16][16];
  int ret[4];
  for(int i=0;i<4;i++){
    ret[i] = 0;
  }
  for(int tile = 0; tile<64; tile++){
    for(int w=0;w<4;w++){
      A_tile_temp[x_local][y_local*4 + w] = A[x*1024 + (tile*16 + y_local*4 + w)];
      B_tile_temp[x_local][y_local*4 + w] = B[(tile*16 + x_local)*1024 + y*4 + w];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k=0;k<16;k++){
      for(int i=0;i<4;i++){
        ret[i] += A_tile_temp[x_local][k]*B_tile_temp[k][y_local*4 + i];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int i=0;i<4;i++){
    C[x*1024 + y*4 + i] = ret[i];
  }
}
