#include <stdlib.h>
#include <iostream>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

#define MAT_SIZE 1024

int data1[MAT_SIZE*MAT_SIZE];
int data2[MAT_SIZE*MAT_SIZE];

int cpu_output[MAT_SIZE*MAT_SIZE];
int gpu_output[MAT_SIZE*MAT_SIZE];

int err;
cl_platform_id platform_id = NULL;
cl_uint nr_platforms;
cl_device_id device_id = NULL;
cl_uint nr_devices;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem memobj_a = NULL;
cl_mem memobj_b = NULL;
cl_mem memobj_c = NULL;

bool gpu_init(void){
  char cBuffer[1024];
  err = clGetPlatformIDs(1, &platform_id, &nr_platforms);
  if(err != CL_SUCCESS){
    cout << "Error getting platform " << endl;
    return false;
  }
  cout << "Number of platforms " << nr_platforms << endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
  if(err != CL_SUCCESS){
    cout << "Error getting platform info" << endl;
    return false;
  }
  cout << "Platform is " << cBuffer << endl;

  err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
  if(err != CL_SUCCESS){
    cout << "Error getting platform version" << endl;
    return false;
  }
  cout << "Platform version is " << cBuffer << endl;

  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &nr_devices);
  if(err != CL_SUCCESS){
    cout << "Error getting device id " << endl;
    return false;
  }
  cout << "Number of devices " << nr_devices << endl;

  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if(err != CL_SUCCESS){
    cout << "Error getting context" << endl;
    return false;
  }
  cout << "Context created" << endl;

  command_queue = clCreateCommandQueue(context, device_id, 0, &err);
  if(err != CL_SUCCESS){
    cout << "Error creating command queue" << endl;
    return false;
  }
  cout << "Command queue created" << endl;

  memobj_a = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, MAT_SIZE*MAT_SIZE*sizeof(int), data1, &err);
  memobj_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, MAT_SIZE*MAT_SIZE*sizeof(int), data2, &err);
  memobj_c = clCreateBuffer(context, CL_MEM_READ_WRITE, MAT_SIZE*MAT_SIZE*sizeof(int), NULL, &err);
  if(err != CL_SUCCESS){
    cout << "Error creating device memory buffer" << endl;
    return false;
  }
  cout << "Memory allocated" << endl;
  return true;
}

void gpu_deinit(void){
  clReleaseMemObject(memobj_a);
  clReleaseMemObject(memobj_b);
  clReleaseMemObject(memobj_c);
  clReleaseContext(context);
}

int main(){
  clock_t temp, cpu_time, gpu_time;

  //Generate input dataset
  srand(3);
  for(int i=0;i<MAT_SIZE;i++){
    for(int j=0;j<MAT_SIZE;j++){
      data1[i*MAT_SIZE + j] = rand()%100;
      data2[i*MAT_SIZE + j] = rand()%100;
    }
  }

  //Running matrix multiplication on cpu
  temp = clock();
  for(int i=0;i<MAT_SIZE;i++){
    for(int j=0;j<MAT_SIZE;j++){
      cpu_output[i*MAT_SIZE + j] = 0;
      for(int k=0;k<MAT_SIZE;k++){
        cpu_output[i*MAT_SIZE + j] += data1[i*MAT_SIZE + k]*data2[k*MAT_SIZE + j];
      }
    }
  }
  cpu_time = (float)(clock()-temp)/(CLOCKS_PER_SEC/1000);
  //Setting up gpu for computation
  if(!gpu_init()){
    cout << "Error while gpu init" << endl;
    return 0;
  }
  FILE *fp;
  fp = fopen("MM_naive.cl", "r");
  if(!fp){
    cout << "Failed to load kernel" << endl;
    return 0;
  }
  fseek(fp, 0, SEEK_END);
  size_t source_size = ftell(fp);
  rewind(fp);
  char* source_str = (char*)malloc(source_size);
  fread(source_str, 1, source_size, fp);
  fclose(fp);
  cout << "File read success, source size is " << source_size << endl;
  cout << source_str << endl;
  cl_program prog  = NULL;
  prog = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
  if(err != CL_SUCCESS){
    cout << "Unable to create program from source" << endl;
    return 0;
  }
  else{
    cout << "Program object created" << endl;
  }
  err = clBuildProgram(prog, 1, &device_id, NULL, NULL, NULL); //fourth arg is compile options
  if(err != CL_SUCCESS){
    cout << "Unable to compile kernel program" << endl;
    return 0;
  }
  else{
    cout << "Program building done" << endl;
  }
  cl_kernel kernel = NULL;
  kernel = clCreateKernel(prog, "MM_naive", &err);
  if(err != CL_SUCCESS){
    cout << "Unable to create kernel object" << endl;
    return 0;
  }
  else{
    cout << "Kernel object created from compiled program" << endl;
  }
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj_a);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobj_b);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobj_c);
  if(err != CL_SUCCESS){
    cout << "Arguments cannot be set" << endl;
    return 0;
  }
  else{
    cout << "Kernel arguments set" << endl;
  }
  size_t localWorkSize[2] = {16, 16};
  size_t globalWorkSize[2] = {MAT_SIZE, MAT_SIZE};
  //Running matrix multiplication on gpu
  temp = clock();
  err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if(err != CL_SUCCESS){
    cout << "Task cannot be enqueued" << endl;
    return 0;
  }
  else{
    cout << "GPU computation done" << endl;
  }
  err = clFinish(command_queue);
  gpu_time = (float)(clock()-temp)/(CLOCKS_PER_SEC/1000);
  //Reading gpu computed Results
  err = clEnqueueReadBuffer(command_queue, memobj_c, CL_TRUE, 0, MAT_SIZE*MAT_SIZE*sizeof(int), gpu_output, 0, NULL, NULL);
  if(err != CL_SUCCESS){
    cout << "Data cannot be read" << endl;
    return 0;
  }
  else{
    cout << "Data read done" << endl;
  }
  gpu_deinit();
  clReleaseKernel(kernel);
  clReleaseProgram(prog);
  clReleaseCommandQueue(command_queue);
  bool check = true;
  for(int i=0;i<MAT_SIZE;i++){
    for(int j=0;j<MAT_SIZE;j++){
      if(cpu_output[i*MAT_SIZE + j]!=gpu_output[i*MAT_SIZE + j]){
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
    /*for(int i=0;i<MAT_SIZE;i++){
      for(int j=0;j<MAT_SIZE;j++){
        cout << "(" <<cpu_output[i*1024 + j] << "," << gpu_output[i*1024 + j] << ") ";
      }
      cout << endl;
    }*/
    cout << "Results does not match" << endl;
  }
  return 0;
}
