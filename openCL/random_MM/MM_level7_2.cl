__kernel void MM_level7_2(__global const int4* A, __global const int4* B, __global int* C){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);

  int x_group = get_group_id(0);
  int y_group = get_group_id(1);

  __local int A_tile_temp[128][32+2];
  __local int B_tile_temp[128][32+2];
  int B_cached[8];
  int A_cached;

  int ret[8][8];
  for(int i=0;i<8;i++){
    for(int j=0;j<8;j++){
      ret[i][j] = 0;
    }
  }

  for(int tile = 0; tile<32; tile++){
    for(int copy_load = 0; copy_load<(16/4); copy_load++){
      int thread_id = x_local*16 + y_local;
      int x_index = ((16*16)/(32/4))*copy_load + (thread_id/(32/4));
      int y_index = (((16*16)*copy_load + thread_id)%(32/4));

      int a_x = x_group*(128) + x_index;
      int a_y = tile*(32/4) + y_index;
      int b_x = y_group*(128) + x_index;
      int b_y = tile*(32/4) + y_index;

      int4 vec_a = A[a_x*(1024/4) + a_y];
      int4 vec_b = B[b_x*(1024/4) + b_y];

      A_tile_temp[x_index][y_index*4+0] = vec_a.x;
      A_tile_temp[x_index][y_index*4+1] = vec_a.y;
      A_tile_temp[x_index][y_index*4+2] = vec_a.z;
      A_tile_temp[x_index][y_index*4+3] = vec_a.w;

      B_tile_temp[x_index][y_index*4+0] = vec_b.x;
      B_tile_temp[x_index][y_index*4+1] = vec_b.y;
      B_tile_temp[x_index][y_index*4+2] = vec_b.z;
      B_tile_temp[x_index][y_index*4+3] = vec_b.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k = 0; k<32; k++){
      for(int i=0;i<8;i++){
        B_cached[i] = B_tile_temp[16*i + y_local][k];
      }
      for(int load_x = 0; load_x<8; load_x++){
        A_cached = A_tile_temp[load_x*16 + x_local][k];
        for(int load_y = 0; load_y<8; load_y++){
          //ret[load_x][load_y]+= A_tile_temp[load_x*16 + x_local][k]*B_tile_temp[k][load_y*16 + y_local];
          ret[load_x][load_y]+= A_cached*B_cached[load_y];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int load_x = 0; load_x<8; load_x++){
    for(int load_y = 0; load_y<8; load_y++){
      C[(x_group*128 + load_x*16 + x_local)*1024 + y_group*128 + load_y*16 + y_local] = ret[load_x][load_y];
    }
  }
}
