#include <iostream>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

int main(int argc, char *argv[]){
  int err;
  cl_uint platforms;
  cl_platform_id platform = NULL;
  char cBuffer[1024];

  err = clGetPlatformIDs(1, &platform, &platforms);
  if(err != CL_SUCCESS){
    cout << "Error getting platform " << endl;
    return 0;
  }
  cout << "Number of platforms " << platforms << endl;

  err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
  if(err != CL_SUCCESS){
    cout << "Error getting platform info" << endl;
    return 0;
  }
  cout << "Platform is " << cBuffer << endl;

  err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
  if(err != CL_SUCCESS){
    cout << "Error getting platform version" << endl;
    return 0;
  }
  cout << "Platform version is " << cBuffer << endl;
  cout << "End of the program " << endl;

  return 0;
}
