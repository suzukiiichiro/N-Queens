
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
//【1】：プラットフォームIDの取得
//cl_int clGetPlatformIDs (cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
//platforms 内に返された cl_platform_id の値は、特定のOpenCLプラットフォームを指定するのに用いることができます。
  status = clGetPlatformIDs(1, &platform, NULL);

  if (status != CL_SUCCESS)
  {
    return 1;
  }

  printf("Determining number of devices on platform.\n");
  cl_uint num_devices;
  //【2】：デバイスIDの取得 
  //cl_int clGetDeviceIDs[1] (cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices)
  //CL_DEVICE_TYPE_ALL  システム上の有効なOpneCLデバイス全てです。
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

  if (status != CL_SUCCESS)
    return 2;

  printf("Getting device IDs.\n");
  cl_device_id * devices = malloc(num_devices * sizeof(cl_device_id));
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

  printf("%d devices detected.\n", num_devices);

  if (status != CL_SUCCESS)
    return 3;
//【3】：コンテキストの作成
//cl_context clCreateContext (const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices,
//OpenCLコンテキストを作成します。OpenCLコンテキストは、1つ以上のデバイスと関連付けられて作成されます。コンテキストは、コマンドキューやメモリ、プログラム、カーネルなどといったオブジェクトを扱うために、また、コンテキスト内の指定された1つ以上のデバイス上でカーネルを実行するために用いられます。
  printf("Creating context.\n");
  cl_context context =
    clCreateContext(NULL, num_devices, devices, NULL, NULL, &status);

  if (status != CL_SUCCESS)
    return 4;

  printf("Loading kernel code.\n");

  char * code;
  //カーネルのコードを読み込む
  get_queens_code(&code);

  if (code == NULL)
  {
    printf("Couldn't load the code.");
  }

  printf("Creating program.\n");
  //【4】：実装・コンパイルしたプログラムを読み込む
  //cl_program clCreateProgramWithSource (cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret)
  //指定したOpenCLコンテキストについてプログラムオブジェクトを作成し、strings 配列内のソースコードをそのプログラムオブジェクト内に読み込みます。context と関連付けられたデバイスがプログラムオブジェクトと関連付けられます。
  cl_program program =
    clCreateProgramWithSource(context, 1, (const char **) &code, NULL, &status);

  free(code);

  if (status != CL_SUCCESS)
    return 5;

  printf("Building program.\n");
  //【5】：読み込んだプログラムをビルドする
  //特定のデバイス用の実行可能プログラムを、プログラムソースもしくはバイナリからビルド（コンパイルとリンク）します。
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
  //【6】：カーネルの作成
  //cl_kernel clCreateKernel (cl_program program, const char *kernel_name, cl_int *errcode_ret)
  //OpenCL の処理を実行するための OpenCL カーネルを生成します。カーネルの概念が分からないときは、カーネルはプログラムとデバイス間をやりとりするためのもの、あるいは、関数と引数がセットになったもの、程度に覚えておけば十分だと思います。
  //対象のプログラムと実行する関数名を引数に指定している
  cl_kernel kernel = clCreateKernel(program, "place", &status);

  if (status != CL_SUCCESS)
    return 1;
  //【7】：コマンドキューの作成
  //OpenCLではCPU側がこの「コマンドキュー」を作成しておいて、作成したコンテキストで使用する
//「各種OpenCLのコマンド」をここに格納しておく仕組みになっている。
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
      //【8】：GPU上にメモリ確保する
      //OpenCLではバッファオブジェクトを経由して、GPUへデータを転送する。
      //そのため、まずはバッファオブジェクトを作成する。
      //cl_mem clCreateBuffer(cl_context context,//OpenCLのコンテキストを指定
      //      cl_mem_flags flags,//フラグ
      //            size_t size, //確保するメモリのバイトサイズ
      //                  void *host_ptr,//CPU側のデータへのポインタ
      //                        cl_int *errcode_ret)//エラーコードを拾う時に使用
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(inProgress), NULL, &status);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't create buffer.\n");

      return 1;
    }
    //【9】：【6】で作成したカーネルの引数に値をセットする
    //■第１引数：kernel
    //値をセットするカーネル。
    //■第２引数：arg_index
    //引数のインデックス。左からひとつめの引数のインデックスを０とする。n個引数を持つカーネルは0..n-1のインデックスの引数を持つ。
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't set kernel arg.");
      return 1;
    }

    //【10】：CPU ⇒ GPU のデータ転送
    //cl_int clEnqueueReadBuffer(cl_command_queue command_queue,//コマンドキュー
    //      cl_mem buffer,//バッファオブジェクト
    //            cl_bool blocking_read,//ブロッキングモードを指定(CL_TRUE or CL_FALSE)
    //                  size_t offset,//データの開始位置のオフセット
    //                        size_t cb,//データのバイトサイズ
    //                              void *ptr,//読み出し先。CPU側の格納用配列データのアドレス
    //                                    cl_uint num_events_in_wait_list,//イベントのリスト。とりあえず「0」でいい
    //                                          const cl_event *event_wait_list,//イベントリストの数。とりあえず「NULL」でいい
    //                                                cl_event *event)//とりあえずNULLでいい
    status = clEnqueueWriteBuffer(cmd_queue, buffer, CL_TRUE, 0, sizeof(inProgress),
 	                                &inProgress, 0, NULL, NULL);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't enque write buffer command.");
      return 1;
    }

    size_t globalSizes[] = { spread };
    //【11】：カーネルの実行
    //カーネル(GPU側の関数)の実行には、「clEnqueueNDRangeKernel関数」を使う。
    //
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

    //【12】：GPU ⇒ CPU のデータ転送
    status = clEnqueueReadBuffer(cmd_queue, buffer, CL_TRUE, 0,
                                 sizeof(inProgress), inProgress, 0, NULL, NULL);

    if (status != CL_SUCCESS)
    {
      printf("Couldn't enque read command.");
      return 1;
    }
    //【13】： 実行が終わるまで待つ
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
