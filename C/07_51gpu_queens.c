/**

   51. GPU(07_37 *N*si アルゴリムは全部のせ) 

	07_46までのロジックを全て含み GPUをNからＮ*siに変更

  struct queenState inProgress[MAX*MAX];

   実行方法
   $ gcc -Wall -W -O3 -std=c99 -pthread -lpthread -lm -o 07_51NQueen 07_51gpu_queens.c -framework OpenCL
   $ ./07_51NQueen 

51. GPU(07_37 *N*si アルゴリムは全部のせ) 
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.02
12:             14200              1787          00:00:00:00.08
13:             73712              9233          00:00:00:00.29
14:            365596             45752          00:00:00:01.24
15:           2279184            285053          00:00:00:06.71

50. GPU(07_36 *N アルゴリムは全部のせ) 
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.01
10:               724                92          00:00:00:00.03
11:              2680               341          00:00:00:00.13
12:             14200              1787          00:00:00:00.44
13:             73712              9233          00:00:00:02.10
14:            365596             45752          00:00:00:11.28
15:           2279184            285053          00:00:01:04.32

49. GPU(07_38 *N*si*si アルゴリムはバックトラック+ビットマップまで) 
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.01
12:             14200                 0          00:00:00:00.05
13:             73712                 0          00:00:00:00.18
14:            365596                 0          00:00:00:00.76
15:           2279184                 0          00:00:00:03.69

48. GPU(07_37 *N*si アルゴリムはバックトラック+ビットマップまで) 
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.01
12:             14200                 0          00:00:00:00.05
13:             73712                 0          00:00:00:00.18
14:            365596                 0          00:00:00:00.91
15:           2279184                 0          00:00:00:05.03

47. GPU(07_36拡張 *N) 
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.01
11:              2680                 0          00:00:00:00.05
12:             14200                 0          00:00:00:00.24
13:             73712                 0          00:00:00:01.13
14:            365596                 0          00:00:00:06.25
15:           2279184                 0          00:00:00:38.69

46. 整理整頓(GPUシングル完成版)
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.01
10:               724                92          00:00:00:00.07
11:              2680               341          00:00:00:00.26
12:             14200              1787          00:00:00:01.11
13:             73712              9233          00:00:00:05.98
14:            365596             45752          00:00:00:34.33

45. 枝刈り (07_09GPU版)
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.01
10:               724                92          00:00:00:00.07
11:              2680               341          00:00:00:00.26
12:             14200              1787          00:00:00:01.12
13:             73712              9233          00:00:00:05.99
14:            365596             45752          00:00:00:34.40

44. クイーンの位置による振り分け(BOUND1 BOUND207_08GPU版)
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.01
 9:               352                46          00:00:00:00.05
10:               724                92          00:00:00:00.18
11:              2680               341          00:00:00:00.58
12:             14200              1787          00:00:00:03.46
13:             73712              9233          00:00:00:18.17
14:            365596             45752          00:00:01:54.16

43. ビットマップ(07_07GPU版)
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.01
 9:               352                46          00:00:00:00.07
10:               724                92          00:00:00:00.25
11:              2680               341          00:00:00:01.15
12:             14200              1787          00:00:00:06.19
13:             73712              9233          00:00:00:34.69
14:            365596             45752          00:00:03:21.72

42. GPU 対称解除法(07_05GPU版)　枝刈り
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.02
 9:               352                46          00:00:00:00.07
10:               724                92          00:00:00:00.29
11:              2680               341          00:00:00:01.36
12:             14200              1787          00:00:00:07.17
13:             73712              9233          00:00:00:39.87
14:            365596             45752          00:00:03:56.84

41. GPU 対称解除法(07_04GPU版)     
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.02
 9:               352                46          00:00:00:00.08
10:               724                92          00:00:00:00.31
11:              2680               341          00:00:00:01.53
12:             14200              1787          00:00:00:08.28
13:             73712              9233          00:00:00:47.21
14:            365596             45752          00:00:04:47.34

40. GPU バックトラック(07_03GPU版)     
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.03
 9:               352                 0          00:00:00:00.12
10:               724                 0          00:00:00:00.43
11:              2680                 0          00:00:00:02.07
12:             14200                 0          00:00:00:11.25
13:             73712                 0          00:00:01:04.73
14:            365596                 0          00:00:06:41.58

*/

#include "stdio.h"
#include "string.h"
#ifdef __APPLE__ // MacOSであるかを確認
#include "OpenCL/cl.h" //MacOSの場合はインクルード
#else
#include<CL/cl.h> //Windows/Unix/Linuxの場合はインクルード
#endif

#define PROGRAM_FILE "./07_51queen_kernel.c" //カーネルソースコード
#define FUNC "place" //カーネル関数の名称を設定
#include "time.h"
#include "sys/time.h"
#define BUFFER_SIZE 4096
#define MAX 27
#define USE_DEBUG 0

cl_device_id *devices;
cl_mem buffer;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue cmd_queue;
cl_platform_id platform;
cl_uint num_devices;

long lGTotal;
long lGUnique;
enum { Place,Remove,Done };
struct HIKISU{
  int Y;
  int I;
  int M;
  int L;
  int D;
  int R;
  int B;
};
struct STACK{
  struct HIKISU param[MAX];
  int current;
};
struct queenState {
  int BOUND1;
  int si;
  int aB[MAX];
  long lTotal; // Number of solutinos found so far.
  int step;
  int y;
  // int startCol; // First column this individual computation was tasked with filling.
  int bm;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  long lUnique; // Number of solutinos found so far.
  int bend;
  int rflg;
  struct STACK stParam;
  int msk;
  int l;
  int d;
  int r;
  int B1;
  int j;
  long lt;
  // long C2;
  // long C4;
  // long C8;
// };
} __attribute__((packed));

struct queenState inProgress[MAX*MAX];
/**
 * カーネルコードの読み込み
 */
void get_queens_code(char **buffer){
  char prefix[256];
  int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n");
  //int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n//#define SIZE %d\n",si);
  //  int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n #define SIZE %d\n",si);
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
プラットフォーム一覧を取得
現在利用可能なOpenCLのプラットフォームの情報を取得
clGetPlatformIDs()使用できるプラットフォームの数とID一覧を取得する関数
numEntries:追加可能なエントリーの数
platforms : 見つかったプラットフォームの一覧が代入されるポインタ
numPlatforms : 使用できるプラットフォームの数が代入されるポインタ  
戻り値　CL_SUCCESS 成功 CL_INVALID_VALUE 失敗
*/
int getPlatform(){
	char value[BUFFER_SIZE];
	size_t size;
  cl_int status;
  status=clGetPlatformIDs(1,&platform,NULL);//pletformは一つでよし
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform ID.");
    return 1; 
  }
	status=clGetPlatformInfo(platform,CL_PLATFORM_NAME,BUFFER_SIZE,value,&size);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform info.");
    return 2; 
  }else{
    if(USE_DEBUG>0) printf("CL_PLATFORM_NAME:%s\n",value);
  }
	status=clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,value,&size);	
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform info.");
    return 3; 
  }else{
    if(USE_DEBUG>0) printf("CL_PLATFORM_VERSION:%s\n",value);
  }
  return 0;
}
/*
デバイス一覧を取得
clGetDeviceIds()使用できるデバイスの数とID一覧を取得する関数
platform : platformを指定
CL_DEVICE_TYPE_CPU：ホストプロセッサを指定する
CL_DEVICE_TYPE_GPU：GPUデバイスを指定する。
CL_DEVICE_TYPE_ACCELERATOR：OpenCL専用デバイスを指定する
CL_DEVICE_TYPE_CUSTOM：OpenCL C言語で実装されたプログラムに対応していないデバイスを指定する。
CL_DEVICE_TYPE_DEFAULT：システム上で設定されているデフォルトのOpenCLデバイス。
CL_DEVICE_TYPE_ALL：C言語で実装されたプすべての使用可能なOpenCLデバイスを指定する。
DEVICE_MAX : 取得するデバイスの制限数。
devices : 見つかったOpenCLデバイスID一覧を取得するためのポインタ。
&deviceCount : 第３引数device_typeに適合したOpenCLデバイスの数を取得するためのポインタ。
*/
int getDeviceID(){
	char value[BUFFER_SIZE];
	size_t size;
  cl_int status;
 	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&num_devices);
	if(status==CL_DEVICE_NOT_FOUND){
 		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&num_devices);
    if(USE_DEBUG>0) printf("CL_DEVICE_TYPE_ALL\n");
	}
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get device count.");
    return 4; 
  }else{
    if(USE_DEBUG>0) printf("CL_DEVICE COUNT:%d\n",num_devices);
  }
  devices=malloc(num_devices * sizeof(cl_device_id));
	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,num_devices,devices,NULL);
	if(status==CL_DEVICE_NOT_FOUND){
		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,num_devices,devices,NULL);
    if(USE_DEBUG>0) printf("CL_DEVICE_TYPE_ALL\n");
	}else{
    if(USE_DEBUG>0) printf("CL_DEVICE_TYPE_GPU\n");
	}
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform device count.");
    return 5; 
  }else{
    if(USE_DEBUG>0) printf("CL_DEVICE INFO\n");
  }
	for(int didx=0;didx<(int)num_devices;didx++){
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_NAME,BUFFER_SIZE,value,&size);	
    if(status!=CL_SUCCESS){
      printf("Couldn't get device name.");
      return 6;
    }else{
      if(USE_DEBUG>0) printf(" +CL_DEVICE_NAME:%s\n",value);
    }
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_VERSION,BUFFER_SIZE,value,&size);
    if(status!=CL_SUCCESS){
      printf("Couldn's get device version.");
      return 7;
    }else{
      if(USE_DEBUG>0) printf("  CL_DEVICE_VERSION:%s\n",value);
    }
    cl_uint units;
    status=clGetDeviceInfo(devices[didx],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(units),&units,NULL);
    if(status!=CL_SUCCESS){
      printf("Couldn's get units in device.");
      return 7;
    }else{
      if(USE_DEBUG>0) printf("  CL_DEVICE_MAX_COMPUTE_UNITS:%d\n",units);
    }
	}
  return 0;
}
/**
コンテキストオブジェクトの作成
clCreateContext()ひとつ以上のデバイスで使用するためのコンテキストを作成する。
nullptr コンテキストプロパティを指定する。
各プロパティ名にはそれぞれに対応した要求される値が続く。この一覧の終端には0がつけ
られる。引数porpertiesには、処理依存のプラットフォームの場合に限りNULLを指定する
ことができる。
num_devices : 対応するデバイスの数
devices : 一意に定まる、clGetDeviceIDs関数で取得されたデバイス
nullptr : アプリケーションによって登録することが可能なコールバック関数。
nullptr : 引数pfn_notifyで設定したコールバック関数が呼び出されたとき、データが
渡されるポインタ。この引数はNULLにした場合、無視される
&err エラーが発生した場合、そのエラーに合わせたエラーコードが返される。
*/
int createContext(){
  cl_int status;
  context=clCreateContext(NULL,num_devices,devices,NULL,NULL,&status);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating context.\n");
    return 8; 
  }else{
    if(USE_DEBUG>0) printf("Creating context.\n");
  }
  return 0;
}
/**
 * カーネルソースの読み込み
 */
int createProgramWithSource(){
  cl_int status;
  char * code;
  get_queens_code(&code);
  if(code==NULL){
    printf("Couldn't load the code.\n");
    return 9;
  }else{
    if(USE_DEBUG>0) printf("Loading kernel code.\n");
  }
  program=clCreateProgramWithSource(context,1,(const char **)&code,NULL,&status);
  free(code);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating program.");
    return 10; 
  }else{
    if(USE_DEBUG>0) printf("Creating program.\n");
  }
  return 0;
}
/**
プログラムのビルド
clBuildProgram()カーネルオブジェクトを作成する。
program    実行ファイルを作成するもとになるプログラム
kernel_name    __kernelで事前に指定する関数名。
kernel_name    __kernelで事前に指定する関数名。
errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
*/
int buildProgram(){
  cl_int status;
  //status=clBuildProgram(program,num_devices,devices,NULL,NULL,NULL);
  status=clBuildProgram(program,1,devices,"",NULL,NULL);
  if(status!=CL_SUCCESS){
    char log[2048];
    size_t logSize;
   	status=clGetProgramBuildInfo(program,devices[0],CL_PROGRAM_BUILD_LOG,2048,log,&logSize);
    if(USE_DEBUG>0) printf("##### ERROR MESSAGE \n%s\n#####\n",log);
    return 11;
  }else{
    if(USE_DEBUG>0) printf("Building program.\n");
  }
  return status;
}
/** 
 * カーネルオブジェクトの宣言
 * 
 */
int createKernel(){
  cl_int status;
  kernel=clCreateKernel(program,FUNC,&status);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating kernel.");
    return 12; 
  }else{
    if(USE_DEBUG>0) printf("Creating kernel.\n");
  }
  return 0;
}
/**
コマンドキューの生成
clCreateCommandQueue()指定したデバイスのコマンドキューを作成する。
context   コンテキスト。
device    第１引数のcontextに関連づけられたデバイス。
properties    コマンドキューに適用するプロパティのリスト。
errcode_ret    エラーコードを格納する変数。
*/
int commandQueue(){
  cl_int status;
#ifdef CL_VERSION_2_0
	cmd_queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, &status);
#else
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cmd_queue = clCreateCommandQueue(context, devices[0], properties, &status);
#endif
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating command queue.");
    return 13; 
  }else{
    if(USE_DEBUG>0)
    printf("Creating command queue.\n");
  }
  return 0;
} 
/**
カーネルオブジェクトをキューに入れる
デバイスメモリを確保しつつデータをコピー
clCreateBuffer()バッファオブジェクトを作成する。
context バッファオブジェクトを作成するために必要なOpenCLコンテキスト。
flags    「バッファオブジェクトをどのようなメモリ領域に割り当てるか」「メモリ領域
をどのように使用するか」のような割り当てやusageに関する情報を指定するビットフィールド。
CL_MEM_READ_WRITE カーネルにメモリ領域へのRead/Writeを許可する設定。
CL_MEM_USE_HOST_PTR デバイスメモリ内でhost_ptrを指定することにより、OpsnCL処理に
バッファをキャッシュすることを許可する。
size    割り当てられたバッファメモリオブジェクトのバイトサイズ
host_ptr    アプリケーションにより既に割り当てられているバッファデータへのポインタ。
errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
*/
void* aligned_malloc(size_t required_bytes, size_t alignment) {
	void* p1; 	// original block
	void** p2; 	// aligned block
	int offset=(int)alignment-1+sizeof(void*);
	if ((p1 = (void*)malloc(required_bytes + offset)) == NULL) {
		 return NULL;
	}
	p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
	p2[-1] = p1;
	return p2;
}
int makeInProgress(int si){
  cl_int status;
	/**************/
  int B2=si-1;
  for(int i=0;i<si;i++){
    for(int j=0;j<si;j++){
      inProgress[i*si+j].BOUND1=i;
      inProgress[i*si+j].si=si;
      for (int m=0;m< si;m++){ inProgress[i*si+j].aB[m]=m;}
      inProgress[i*si+j].lTotal=0;
      inProgress[i*si+j].step=0;
      inProgress[i*si+j].y=0;
      inProgress[i*si+j].bm=0;
      inProgress[i*si+j].BOUND2=B2;
      inProgress[i*si+j].ENDBIT=0;
      inProgress[i*si+j].TOPBIT=1<<(si-1);
      inProgress[i*si+j].SIDEMASK=0;
      inProgress[i*si+j].LASTMASK=0;
      inProgress[i*si+j].lUnique=0;
      inProgress[i*si+j].bend=0;
      inProgress[i*si+j].rflg=0;
      for (int m=0;m<si;m++){
        inProgress[i*si+j].stParam.param[m].Y=0;
        inProgress[i*si+j].stParam.param[m].I=si;
        inProgress[i*si+j].stParam.param[m].M=0;
        inProgress[i*si+j].stParam.param[m].L=0;
        inProgress[i*si+j].stParam.param[m].D=0;
        inProgress[i*si+j].stParam.param[m].R=0;
        inProgress[i*si+j].stParam.param[m].B=0;
      }
      inProgress[i*si+j].stParam.current=0;
      inProgress[i*si+j].msk=(1<<si)-1;
      inProgress[i*si+j].l=0;
      inProgress[i*si+j].d=0;
      inProgress[i*si+j].r=0;
      inProgress[i*si+j].B1=0;
      inProgress[i*si+j].j=j;
      inProgress[i*si+j].lt=0;
      // inProgress[i*si+j].C2=0;
      // inProgress[i*si+j].C4=0;
      // inProgress[i*si+j].C8=0;
    }
      B2--;
  }
	/**************/
  if(USE_DEBUG>0) printf("Starting computation of Q(%d)\n",si);
  buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(inProgress), NULL, &status);
  clRetainMemObject(buffer);
  if(USE_DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't create buffer.\n"); return 14; }
  /**
   *
   *
   */
	status=clEnqueueWriteBuffer(cmd_queue,buffer,CL_FALSE,0,sizeof(inProgress),&inProgress,0,NULL,NULL); 
  if(USE_DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't enque write buffer command."); return 16; }

  /**
   * マップオブジェクトの解放
   */
/**
  status = clEnqueueUnmapMemObject(
        cmd_queue,  //投入キュー
        buffer,     //対象のOpenCLバッファ
        ptrMappedA, //取得したホスト側のポインタ
        0,          //この関数が待機すべきeventの数
        NULL,       //この関数が待機すべき関数のリストへのポインタ
        NULL);      //この関数の返すevent
  if(USE_DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't finish command queue."); return 14; }
*/
 /**
    カーネルの引数をセット
    clSetKernelArg()カーネルの特定の引数に値をセットする。
    kernel    値をセットするカーネル。
    arg_index    引数のインデックス。
    arg_size    引数として渡すのデータのサイズ。
    arg_value    第２引数arg_indexで指定した引数にわたすデータへのポインタ。
  */
  status=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
  if(USE_DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't set kernel arg."); return 15; }
  return 0;
}
/**
 * タスクの終了を待機する
 */
int all_tasks_done(int32_t num_tasks) {
	for (int i=0;i<num_tasks;i++){
    // for (int j=0;j<num_tasks;j++){
      //if (inProgress[i*num_tasks+j].step != 2){
      if (inProgress[i].step != 2){
        // printf("notfinish:i:%d:step:%d\n",i*num_tasks+j,inProgress[i*num_tasks+j].msk);
        return 0;
      }  
    // }
  }
	return 1;
}
/**
  カーネルの実行 
  カーネルを実行するコマンドをキューに入れて、カーネル関数をデバイスで実行
  work sizeの指定
  ここでは1要素に対して1 work item
  またグループあたり1 work item (実は効率的でない)
  width * heightの2次元でwork itemを作成
*/
int execKernel(int si){
  cl_int status;
	/**************/
  while(!all_tasks_done(si*si)){
    //size_t dim=1;
    cl_uint dim=1;
    size_t globalWorkSize[] = {si*si};
	/**************/
    size_t localWorkSize[] = { 1 };
    status=clEnqueueNDRangeKernel(
        cmd_queue,         //タスクを投入するキュー
        kernel,            //実行するカーネル
        dim,               //work sizeの次元
        NULL,              //NULLを指定すること
        globalWorkSize,    //全スレッド数
        localWorkSize,     //1グループのスレッド数
        0,                 //この関数が待機すべきeventの数
        NULL,              //この関数が待機すべき関数のリストへのポインタ
        NULL);             //この関数の返すevent
    if(USE_DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't enque kernel execution command."); return 17; }
    /**
     * 結果を読み込み
     */
    status=clEnqueueReadBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
    if(USE_DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't enque read command."); return 18; }
 } //end while
  return 0;
}
/**
 * 結果の印字
 *
 */
int execPrint(int si){
  lGTotal=0;
  lGUnique=0;
	/**************/
  for(int i=0;i<si;i++){
    for(int j=0;j<si;j++){
          // if(USE_DEBUG>0) printf("lTotal:%ld\n",inProgress[i*si+j].lTotal);
          // printf("BOUND1:%d\n",inProgress[i*si+j].BOUND1);
          // printf("si:%d\n",inProgress[i*si+j].si);
          //printf("a:step:%d\n",inProgress[i*si+j].step);
          //printf("msk:step:%d\n",inProgress[i*si+j].msk);
          // printf("y:%d\n",inProgress[i*si+j].y);
          // printf("bm:%d\n",inProgress[i*si+j].bm);
          // printf("BOUND2:%d\n",inProgress[i*si+j].BOUND2);
          // printf("ENDBIT:%d\n",inProgress[i*si+j].ENDBIT);
          // printf("TOPBIT:%d\n",inProgress[i*si+j].TOPBIT);
          // printf("SIDEMASK:%d\n",inProgress[i*si+j].SIDEMASK);
          // printf("LASTMASK:%d\n",inProgress[i*si+j].LASTMASK);
          // printf("bend:%d\n",inProgress[i*si+j].bend);
          // printf("rflg:%d\n",inProgress[i*si+j].rflg);
          // printf("l:%d\n",inProgress[i*si+j].l);
          // printf("d:%d\n",inProgress[i*si+j].d);
          // printf("r:%d\n",inProgress[i*si+j].r);
          // printf("B1:%d\n",inProgress[i*si+j].B1);
          // printf("j:%d\n",inProgress[i*si+j].j);
          // printf("lUnique:%ld\n",inProgress[i*si+j].lUnique);
           // printf("lt:%ld\n",inProgress[i*si+j].lt);
          // lGTotal+=inProgress[i*si+j].lt;
          // lGUnique+=inProgress[i*si+j].lUnique;
        // lGTotal+=inProgress[i*si+j].C2*2+inProgress[i*si+j].C4*4+inProgress[i*si+j].C8*8;
        lGTotal+=inProgress[i*si+j].lTotal;
        lGUnique+=inProgress[i*si+j].lUnique;
        // lGUnique+=inProgress[i*si+j].C2+inProgress[i*si+j].C4+inProgress[i*si+j].C8;
    }
  }
	/**************/
  return 0;
}
int create(){
  int rst;
  while (1){
    createProgramWithSource();  // ソースコードからカーネルプログラム作成
    rst=buildProgram();             // カーネルプログラムのビルド
    if(rst==0){
      break;
    }
  }
  return 0;
}
/**
 * clGetProgramBuildInfo();		// プログラムのビルド情報を取得
 *
*/
int NQueens(int si){
  struct timeval t0; struct timeval t1; int ss;int ms;int dd;
  makeInProgress(si);
  gettimeofday(&t0,NULL);    // 計測開始
  execKernel(si);
  gettimeofday(&t1,NULL);    // 計測終了
  execPrint(si);
	clReleaseMemObject(buffer);
  clReleaseContext(context);
  if (t1.tv_usec<t0.tv_usec) {
    dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
    ss=(t1.tv_sec-t0.tv_sec-1)%86400;
    ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
  } else { 
    dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
    ss=(t1.tv_sec-t0.tv_sec)%86400;
    ms=(t1.tv_usec-t0.tv_usec+500)/10000;
  }
  int hh=ss/3600;
  int mm=(ss-hh*3600)/60;
  ss%=60;
  printf("%2d:%18llu%18llu%12.2d:%02d:%02d:%02d.%02d\n", si,(unsigned long long)lGTotal,(unsigned long long)lGUnique,dd,hh,mm,ss,ms);
  return 0;
}
/**
 *
 *
 */
int main(void){
  int min=4;
  int targetN=19;
  //Nが変化しても変動のないメソッドを１回だけ実行
  getPlatform();              // プラットフォーム一覧を取得
  getDeviceID();              // デバイス一覧を取得
  createContext();            // コンテキストの作成
  commandQueue();             // コマンドキュー作成
  create();
  createKernel();             // カーネルの作成
  printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=targetN;i++){
    lGTotal=0; 
    lGUnique=0;
    NQueens(i); //スレッド実行
  }
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(cmd_queue);
  free(devices);
  return 0;
}
