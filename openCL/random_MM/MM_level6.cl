__kernel void MM_level6(__global const int* A, __global const int* B, __global int* C){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);

  int x_group = get_group_id(0);
  int y_group = get_group_id(1);

  __local int A_tile_temp[128][32+2];
  __local int B_tile_temp[32][128];
  /*int B_cached[128];
  int A_cached;*/

  int ret[8][8];
  for(int i=0;i<8;i++){
    for(int j=0;j<8;j++){
      ret[i][j] = 0;
    }
  }

  for(int tile = 0; tile<32; tile++){
    for(int copy_load = 0; copy_load<16; copy_load++){
      int thread_id = x_local*16 + y_local;
      int x_index = ((16*16)/32)*copy_load + (thread_id/32);
      int y_index = ((16*16)*copy_load + thread_id )%32;

      int a_x = x_group*128 + x_index;
      int a_y = tile*32 + y_index;
      int b_x = y_group*128 + x_index;
      int b_y = tile*32 + y_index;

      A_tile_temp[x_index][y_index] = A[a_x*1024 + a_y];
      B_tile_temp[y_index][x_index] = B[b_x*1024 + b_y];
    }
    //Alternate copy method
    /*for(int i=0;i<8;i++){
      for(int j=0;j<2;j++){
        int a_x = x_group*128 + i*16+x_local;
        int a_y = tile*32 + j*16+y_local;
        int b_x = y_group*128 + i*16+x_local;
        int b_y = tile*32 + j*16+y_local;
        A_tile_temp[i*16+x_local][j*16+y_local] = A[a_x*1024+a_y];
        B_tile_temp[j*16+y_local][i*16+x_local] = B[b_x*1024+b_y];
      }
    }*/
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k = 0; k<32; k++){
      /*for(int i=0;i<128;i++){
        B_cached[i] = B_tile_temp[k][i];
      }*/
      for(int load_x = 0; load_x<8; load_x++){
        //A_cached = A_tile_temp[load_x*16 + x_local][k];
        for(int load_y = 0; load_y<8; load_y++){
          ret[load_x][load_y]+= A_tile_temp[load_x*16 + x_local][k]*B_tile_temp[k][load_y*16 + y_local];
          //ret[load_x][load_y]+= A_cached*B_cached[load_y*16 + y_local];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int load_x = 0; load_x<8; load_x++){
    for(int load_y = 0; load_y<8; load_y++){
      C[(x_group*128 + load_x*16 + x_local)*1024 + y_group*128 + load_y*16 + y_local] = ret[load_x][load_y];
    }
  }
}
