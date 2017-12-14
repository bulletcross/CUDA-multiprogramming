#include <stdlib.h>
#include <iostream>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;


int N = 1024;
int M = 1024;
#define N_size 1024
#define M_size 1024
int input[N_size*M_size];
int gpu_output[M_size*N_size];
int cpu_output[M_size*N_size];

int err;
cl_platform_id platform_id = NULL;
cl_uint nr_platforms;
cl_device_id device_id = NULL;
cl_uint nr_devices;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem memobj_a = NULL;
cl_mem memobj_b = NULL;

cl_program prog  = NULL;
cl_kernel kernel = NULL;

bool gpu_init(void){
  char cBuffer[1024];
  err = clGetPlatformIDs(1, &platform_id, &nr_platforms);
  if(err != CL_SUCCESS){
    cout << "Error getting platform " << err << endl;
    return false;
  }
  cout << "Number of platforms " << nr_platforms << endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
  if(err != CL_SUCCESS){
    cout << "Error getting platform info " << err <<endl;
    return false;
  }
  cout << "Platform is " << cBuffer << endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
  if(err != CL_SUCCESS){
    cout << "Error getting platform version " << err << endl;
    return false;
  }
  cout << "Platform version is " << cBuffer << endl;

  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &nr_devices);
  if(err != CL_SUCCESS){
    cout << "Error getting device id " << err << endl;
    return false;
  }
  cout << "Number of devices " << nr_devices << endl;

  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if(err != CL_SUCCESS){
    cout << "Error getting context " << err << endl;
    return false;
  }
  cout << "Context created" << endl;

  command_queue = clCreateCommandQueue(context, device_id, 0, &err);
  if(err != CL_SUCCESS){
    cout << "Error creating command queue " << err << endl;
    return false;
  }
  cout << "Command queue created" << endl;
  memobj_a = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N*M*sizeof(int), input, &err);
  memobj_b = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(int), NULL, &err);
  if(err != CL_SUCCESS){
    cout << "Error creating device memory buffer " << err << endl;
    return false;
  }
  cout << "Memory allocated" << endl;
  return true;
}

bool setup_program(string name){
  FILE *fp;
  fp = fopen((name+".cl").c_str(), "r");
  if(!fp){
    cout << "Failed to load kernel" << endl;
    return false;
  }
  fseek(fp, 0, SEEK_END);
  size_t source_size = ftell(fp);
  rewind(fp);
  char* source_str = (char*)malloc(source_size);
  fread(source_str, 1, source_size, fp);
  fclose(fp);
  cout << "File read success, source size is " << source_size << endl;
  cout << source_str << endl;

  prog = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
  if(err != CL_SUCCESS){
    cout << "Unable to create program from source " << err << endl;
    return false;
  }
  else{
    cout << "Program object created" << endl;
  }
  err = clBuildProgram(prog, 1, &device_id, NULL, NULL, NULL);
  if(err != CL_SUCCESS){
    cout << "Unable to compile kernel program " << err << endl;
    char *log = new char[1024];
    err = clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, 1024, log, NULL);
    cout << log << endl;
    return false;
  }
  else{
    cout << "Program building done" << endl;
    size_t bin_size;
    err = clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
    char *bin = new char[bin_size];
    err = clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
    fp = fopen((name + ".ptx").c_str(), "wb");
    fwrite(bin, sizeof(char), bin_size, fp);
    fclose(fp);
    free(bin);
  }
  kernel = clCreateKernel(prog, name.c_str(), &err);
  if(err != CL_SUCCESS){
    cout << "Unable to create kernel object " << err << endl;
    return false;
  }
  else{
    cout << "Kernel object created from compiled program" << endl;
  }
  return true;
}

void gpu_deinit(void){
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(prog);
  clReleaseKernel(kernel);
  clReleaseMemObject(memobj_a);
  clReleaseMemObject(memobj_b);
  clReleaseContext(context);
}

int main(){
  clock_t temp, cpu_time, gpu_time;

  //Generate input dataset
  srand(3);
  for(int i=0;i<N;i++){
    for(int j=0;j<M;j++){
      input[i*M + j] = rand()%100;
    }
  }

  //Performing transpose on cpu
  temp = clock();
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      cpu_output[i*N + j] = input[j*M + i];
    }
  }
  cpu_time = (float)(clock()-temp);

  //Setting up gpu for computation
  if(!gpu_init()){
    cout << "Error while gpu init" << endl;
    return 0;
  }
  //Setup Program
  if(!setup_program("trans_level2")){
    cout << "Could not setup program" << endl;
    return 0;
  }

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj_a);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobj_b);
  err = clSetKernelArg(kernel, 2, sizeof(int), (void *)&N);
  err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&M);

  if(err != CL_SUCCESS){
    cout << "Arguments cannot be set " << err << endl;
    return 0;
  }
  else{
    cout << "Kernel arguments set" << endl;
  }

  /*int nr_tile_x = 128;
  int nr_tile_y = 128;*/
  size_t localWorkSize[2] = {32, 32};
  size_t globalWorkSize[2] = {1024, 1024};
  //Performing transpose on gpu
  temp = clock();
  err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if(err != CL_SUCCESS){
    cout << "Task cannot be enqueued " << err << endl;
    return 0;
  }
  else{
    cout << "GPU computation done" << endl;
  }
  err = clFinish(command_queue);
  gpu_time = (float)(clock()-temp);
  //Reading gpu computed Results
  err = clEnqueueReadBuffer(command_queue, memobj_b, CL_TRUE, 0, M*N*sizeof(int), gpu_output, 0, NULL, NULL);
  if(err != CL_SUCCESS){
    cout << "Data cannot be read " << err << endl;
    return 0;
  }
  else{
    cout << "Data read done" << endl;
  }
  gpu_deinit();

  bool check = true;
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      if(cpu_output[i*N + j]!=gpu_output[i*N + j]){
        check = false;
        break;
      }
    }
    if(!check){
      break;
    }
  }

  if(check){
    cout << "CPU TIME: " << cpu_time << "  GPU TIME: " << gpu_time << endl;
  }
  else{
    for(int i=0;i<M;i++){
      for(int j=0;j<N;j++){
        cout << "(" <<cpu_output[i*N + j] << "," << gpu_output[i*N + j] << ") ";
      }
      cout << endl;
    }
    cout << "Results does not match" << endl;
  }
  return 0;
}
