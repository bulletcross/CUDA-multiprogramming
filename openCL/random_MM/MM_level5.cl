__kernel void MM_level5(__global const int* A, __global const int* B, __global int* C){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);
  int x_group = get_group_id(0);
  int y_group = get_group_id(1);

  __local int A_tile_temp[32][16];
  __local int B_tile_temp[16][32];
  int ret[2];
  for(int i=0;i<2;i++){
    ret[i] = 0;
  }

  for(int tile = 0; tile<64; tile++){
    for(int load = 0; load<2; load++){
      int a_x = x_group*32 + x_local;
      int a_y = tile*16 + y_local*load;
      int b_x = y_group*32 + x_local;
      int b_y = tile*16 + y_local*load;
      A_tile_temp[x_local][y_local + load*8] = A[a_x*1024 + a_y];
      B_tile_temp[y_local + load*8 + x_local] = B[b_x*1024 + b_y];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k = 0; k<16; k++){
      for(int load = 0; load<2; load++){
        ret[load]+= A_tile_temp[x_local][k]*B_tile_temp[k][y_local + load*16];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int load = 0; load<2; load++){
    C[x*1024 + y_group*32 + load*16 + y_local] = ret[load];
  }
}
