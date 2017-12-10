__kernel void MM_level4(__global const int4* A, __global const int4* B, __global int4 *C){
  int x = get_global_id(0);
  int y = get_global_id(1);
  int x_local = get_local_id(0);
  int y_local = get_local_id(1);
  __local int4 A_tile_temp[16][4];
  __local int4 B_tile_temp[16][4];
  int4 ret = {0, 0, 0, 0};
  for(int tile = 0; tile<64; tile++){
    A_tile_temp[x_local][y_local] = A[x*(1024/4) + tile*(16/4) + y_local];
    B_tile_temp[x_local][y_local] = B[(tile*16 + x_local)*(1024/4) + y];
    barrier(CLK_LOCAL_MEM_FENCE);
    int4 vec_a, vec_b;
    int val_a;
    for(int k=0;k<4;k++){
      vec_a = A_tile_temp[x_local][k];
      for(int w = 0; w<4; w++){
        vec_b = B_tile_temp[k*4 + w][y_local];
        switch(w){
          case 0: val_a = vec_a.x; break;
          case 1: val_a = vec_a.y; break;
          case 2: val_a = vec_a.z; break;
          case 3: val_a = vec_a.w; break;
        }
        ret.x += val_a*vec_b.x;
        ret.y += val_a*vec_b.y;
        ret.z += val_a*vec_b.z;
        ret.w += val_a*vec_b.w;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[x*(1024/4) + y] = ret;
}
