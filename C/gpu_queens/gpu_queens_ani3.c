/**
 N:        Total       Unique        hh:mm:ss.ms
 2:            0               0            0.00
 3:            0               0            0.00
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.30
14:       365596               0            1.93
15:      2279184               0           13.50
16:     14772512               0         1:39.30
17:     95815104               0        12:29.59
*/

#include "stdio.h"
#include "string.h"
#ifdef __APPLE__ // MacOSであるかを確認
#include "OpenCL/cl.h" //MacOSの場合はインクルード
#else
#include<CL/cl.h> //Windows/Unix/Linuxの場合はインクルード
#endif

#define PROGRAM_FILE "./queen_kernel.c" //カーネルソースコード
#define FUNC "place" //カーネル関数の名称を設定
#include "time.h"
#include "sys/time.h"
#define BUFFER_SIZE 4096
#define MAX 27
//#define SPREAD 1024
//
//const int32_t si=13;
//const int si=13;
//int si;
//const int si=14;
//const int si=15;
long lGTotal;
long lGUnique;
//typedef int64_t qint;
enum { Place,Remove,Done };
struct queenState {
  int si;
	int BOUND1;
  int id;
  int aB[MAX];
  int step;
  int y;
  int startCol;
  int msk;
  int bm;
  int down;
  int right;
  int left;
  long lTotal;
} __attribute__((packed));
void get_queens_code(char ** buffer,int si);
//int all_tasks_done(struct queenState * tasks,size_t num_tasks);
int all_tasks_done(struct queenState *tasks,int num_tasks);
/**
main()OpenCL 主な流れ 
	clGetPlatformIDs();        // プラットフォーム一覧を取得
	clGetDeviceIDs();          // デバイス一覧を取得
	clCreateContext();         // コンテキストの作成
	clCreateCommandQueue();    // コマンドキュー作成
	clCreateProgramWithSource();// ソースコードからカーネルプログラム作成
	clBuildProgram();						// カーネルプログラムのビルド
	clGetProgramBuildInfo();		// プログラムのビルド情報を取得
	clCreateKernel();						// カーネルの作成
	clCreateBuffer();          // 書き込み・読み込みメモリバッファの作成
	clSetKernelArg();          // カーネル引数の設定
	clEnqueueWriteBuffer();    // メモリバッファへの書き込み
	clEnqueueNDRangeKernel();  // カーネル実行
	clEnqueueReadBuffer();     // メモリバッファから結果読み出し
	clFinish();                // 実行が終わるまで待機
	clReleaseMemObject();
	clReleaseKernel();
	clReleaseProgram();
	clReleaseCommandQueue();
	clReleaseContext();
*/
/**
 * カーネルコードの読み込み
 */
void get_queens_code(char **buffer,int si){
  char prefix[256];
  int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n// #define SIZE %d\n",si);
  FILE * f=fopen(PROGRAM_FILE,"rb");
  if(!f){ *buffer=NULL;return;}
  long fileLength=0; fseek(f,0,SEEK_END); fileLength=ftell(f); fseek(f,0,SEEK_SET);
  long totalLength=prefixLength + fileLength + 1;
  *buffer=malloc(totalLength); strcpy(*buffer,prefix);
  if(buffer){ fread(*buffer + prefixLength,1,fileLength,f);} fclose(f);
  // Replace BOM with space
 (*buffer)[prefixLength]=' '; (*buffer)[prefixLength + 1]=' '; (*buffer)[prefixLength + 2]=' ';
}
/**
 * タスクの終了を待機する
 */
//int all_tasks_done(struct queenState * tasks,size_t num_tasks){
int all_tasks_done(struct queenState *tasks,int num_tasks){
  for(int i=0;i<num_tasks;i++){ if(tasks[i].step==Done){ return 1; } }
  return 0;
}
int NQueens(int si){
  char value[BUFFER_SIZE];
  size_t size;
  cl_platform_id platform;	//プラットフォーム
  cl_uint num_devices;			//デバイス数
  cl_context context;				//コンテキスト
  cl_program program;				//プログラム
  cl_command_queue cmd_queue;//コマンドキュー
  cl_kernel kernel;					//カーネル
  cl_int status=clGetPlatformIDs(1,&platform,NULL);//プラットフォームを取得
  status=clGetPlatformInfo(platform,CL_PLATFORM_NAME,BUFFER_SIZE,value,&size);
  printf("CL_PLATFORM_NAME:%s\n",value);
  status=clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,value,&size);	
  printf("CL_PLATFORM_VERSION:%s\n",value);
  if(status!=CL_SUCCESS){ printf("Couldn't get platform info.");return 1;}
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

  if (status != CL_SUCCESS)
    return 2;
  cl_device_id * devices=malloc(num_devices*sizeof(cl_device_id));
  status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,num_devices,devices,&num_devices);
  if(status==CL_DEVICE_NOT_FOUND){
    status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,num_devices,devices,&num_devices);
    printf("CL_DEVICE_TYPE_ALL\n");
  }else{ printf("CL_DEVICE_TYPE_GPU\n");}
  if(status!=CL_SUCCESS){ printf("Couldn't get device count.");return 2;
  }else{ printf("CL_DEVICE COUNT:%d\n",num_devices);}
  context=clCreateContext(NULL,num_devices,devices,NULL,NULL,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating context.\n");return 4;
  }else{ printf("Creating context.\n");}
  char *code;
  get_queens_code(&code,si);
  if(code==NULL){ printf("Couldn't load the code.");return 5;
  }else{ printf("Loading kernel code.\n");}
  program=clCreateProgramWithSource(context,1,(const char **)&code,NULL,&status);
  free(code);
  if(status!=CL_SUCCESS){ printf("Couldn't creating program.");return 6;
  }else{ printf("Creating program.\n");}
  status=clBuildProgram(program,num_devices,devices,NULL,NULL,NULL);
  if(status!=CL_SUCCESS){
    char log[2048];
    status=clGetProgramBuildInfo(program,devices[0],CL_PROGRAM_BUILD_LOG,2048,log,NULL);
    printf("%s",log);printf("Couldn't building program.");return 7;
  }else{ printf("Building program.\n");}
  kernel=clCreateKernel(program,FUNC,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating kernel.");return 8;
  }else{ printf("Creating kernel.\n");}
  cmd_queue=clCreateCommandQueue(context,devices[0],0,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating command queue.");return 5;
  }else{ printf("Creating command queue.\n");}
  /**
   * 初期化 カーネルオブジェクトに渡す構造体
   */
  struct queenState inProgress[si];
  for(int i=0;i<si;i++){
    struct queenState s;
    s.si=si;
    s.BOUND1=i;//BOUND1の初期化
    s.id=i;
    for (int i=0;i< si;i++){ s.aB[i]=i;}
    s.step=0;
    s.y=0;
    s.startCol =1;
    s.msk=(1<<si)-1;
    s.bm=(1<<si)-1;
    s.down=0;
    s.right=0;
    s.left=0;
    s.lTotal=0;
    inProgress[i]=s;
  }
  printf("Starting computation of Q(%d)\n",si);
  while(!all_tasks_done(inProgress,si)){
    printf("loop\n");
    cl_mem buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(inProgress),NULL,&status);
    if(status!=CL_SUCCESS){ printf("Couldn't create buffer.\n");return 9;}
    status=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
    if(status!=CL_SUCCESS){ printf("Couldn't set kernel arg.");return 11;}
    status=clEnqueueWriteBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
    if(status!=CL_SUCCESS){ printf("Couldn't enque write buffer command.");return 10;}
    size_t globalSizes[]={ si };
    status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
    if(status!=CL_SUCCESS){ printf("Couldn't enque kernel execution command.");return 12;}
    status=clEnqueueReadBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
    if(status!=CL_SUCCESS){ printf("Couldn't enque read command.");return 13;}
    status=clFinish(cmd_queue);
    if(status!=CL_SUCCESS){ printf("Couldn't finish command queue.");return 14;}
  }//end while
  lGTotal=0;
  lGUnique=0;
  for(int i=0;i<si;i++){
    printf("%d: %ld\n",inProgress[i].id,inProgress[i].lTotal);
    lGTotal+=inProgress[i].lTotal;
  }
  free(devices);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);
  return 0;
}
int main(void){
  int min=2;
  struct timeval t0;
  struct timeval t1;
  printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  //for(int i=min;i<=MAX;i++){
  for(int i=14;i<=14;i++){
//    si=i;
    lGTotal=0; lGUnique=0;
    gettimeofday(&t0, NULL);
    NQueens(i);
    gettimeofday(&t1, NULL);
    int ss;int ms;int dd;
    if (t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    } else { 
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
    printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", i,lGTotal,lGUnique,dd,hh,mm,ss,ms);
  }
  return 0;
}
