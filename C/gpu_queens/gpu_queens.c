
#include "stdio.h"
#include "string.h"
#include "OpenCL/cl.h"

const int32_t numQueens = 8;
const int32_t spread = 1024;

typedef int64_t qint;

enum { Place, Remove, Done };

struct queenState
{
  int id;
  qint masks[numQueens];
  uint64_t solutions;
  char step;
  char col;
  char startCol;
  qint mask;
  qint rook;
  qint add;
  qint sub;
} __attribute__((packed));

void get_queens_code(char ** buffer)
{
  char prefix[256];
  int prefixLength =
    snprintf(prefix, 256, "#define OPENCL_STYLE\n#define NUM_QUEENS %d\n", numQueens);

  FILE * f = fopen("./queen_kernel.c", "rb");

  if (!f)
  {
    *buffer = NULL;
    return;
  }

  long fileLength = 0;
  fseek(f, 0, SEEK_END);
  fileLength = ftell(f);
  fseek(f, 0, SEEK_SET);

  long totalLength = prefixLength + fileLength + 1;
  *buffer = malloc(totalLength);

  strcpy(*buffer, prefix);

  if (buffer)
    fread(*buffer + prefixLength, 1, fileLength, f);

  fclose(f);

  // Replace BOM with space
  (*buffer)[prefixLength] = ' ';
  (*buffer)[prefixLength + 1] = ' ';
  (*buffer)[prefixLength + 2] = ' ';
}

int all_tasks_done(struct queenState * tasks, size_t num_tasks)
{
  for (int i = 0; i < num_tasks; i++)
    if (tasks[i].step == Done)
      return 1;

  return 0;
}

int main()
{
  cl_int status;

  printf("Looking up first platform.\n");
  cl_platform_id platform;
  status = clGetPlatformIDs(1, &platform, NULL);

  if (status != CL_SUCCESS)
  {
    return 1;
  }

  printf("Determining number of devices on platform.\n");
  cl_uint num_devices;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

  if (status != CL_SUCCESS)
    return 2;

  printf("Getting device IDs.\n");
  cl_device_id * devices = malloc(num_devices * sizeof(cl_device_id));
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

  printf("%d devices detected.\n", num_devices);

  if (status != CL_SUCCESS)
    return 3;

  printf("Creating context.\n");
  cl_context context =
    clCreateContext(NULL, num_devices, devices, NULL, NULL, &status);

  if (status != CL_SUCCESS)
    return 4;

  printf("Loading kernel code.\n");

  char * code;
  get_queens_code(&code);

  if (code == NULL)
  {
    printf("Couldn't load the code.");
  }

  printf("Creating program.\n");
  cl_program program =
    clCreateProgramWithSource(context, 1, (const char **) &code, NULL, &status);

  free(code);

  if (status != CL_SUCCESS)
    return 5;

  printf("Building program.\n");
  status = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);

  if (status != CL_SUCCESS)
  {
    char log[2048];
    status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
                                   2048, log, NULL);

    printf("%s", log);
    return 6;
  }

  printf("Creating kernel.\n");
  cl_kernel kernel = clCreateKernel(program, "place", &status);

  if (status != CL_SUCCESS)
    return 1;

  printf("Creating command queue.\n");
  cl_command_queue cmd_queue =
    clCreateCommandQueue(context, devices[1], 0, &status);

  if (status != CL_SUCCESS)
    return 1;

  // List of in-progress tasks
  struct queenState inProgress[spread] = { 0 };

  for (int i = 0; i < spread; i++)
  {
    struct queenState s = { 0 };
    s.id = i;
    s.mask = (1 << numQueens) - 1;
    inProgress[i] = s;
  }

  printf("Starting computation of Q(%d)\n", numQueens);

  while (!all_tasks_done(inProgress, spread))
  {
    printf("loop\n");
    
    cl_mem buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(inProgress), NULL, &status);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't create buffer.\n");

      return 1;
    }

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't set kernel arg.");
      return 1;
    }

    status = clEnqueueWriteBuffer(cmd_queue, buffer, CL_TRUE, 0, sizeof(inProgress),
 	                                &inProgress, 0, NULL, NULL);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't enque write buffer command.");
      return 1;
    }

    size_t globalSizes[] = { spread };
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't enque kernel execution command.");
      return 1;
    }

    status = clEnqueueReadBuffer(cmd_queue, buffer, CL_TRUE, 0,
                                 sizeof(inProgress), inProgress, 0, NULL, NULL);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't enque read command.");
      return 1;
    }

    status = clFinish(cmd_queue);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't finish command queue.");
      return 1;
    }
  }

  for (int i = 0; i < spread; i++)
    printf("%d: %llu\n", inProgress[i].id, inProgress[i].solutions);

  free(devices);

  return 0;
}
