/**

   37. GPU ３段スレッド(si*si*si)       NQueen37() N17= 00:35.33 N18=

   実行方法
   $ gcc -Wall -W -O3 -std=c99 -pthread -lpthread -lm -o 07_37NQueen 07_37gpu_queens.c -framework OpenCL
   $ ./07_37NQueen 

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.00
12:             14200                 0          00:00:00:00.01
13:             73712                 0          00:00:00:00.05
14:            365596                 0          00:00:00:00.21
15:           2279184                 0          00:00:00:01.04
16:          14772512                 0          00:00:00:06.26
17:          95815104                 0          00:00:00:47.33
*/

#include "stdio.h"
#include "string.h"
#ifdef __APPLE__ // MacOSであるかを確認
#include "OpenCL/cl.h" //MacOSの場合はインクルード
#else
#include<CL/cl.h> //Windows/Unix/Linuxの場合はインクルード
#endif

#define PROGRAM_FILE "./07_38queen_kernel.c" //カーネルソースコード
#define FUNC "place" //カーネル関数の名称を設定
#include "time.h"
#include "sys/time.h"
#define BUFFER_SIZE 4096
#define MAX 27
#define DEBUG 1
//#define SPREAD 1024
//
//const int32_t si=13;
//const int si=13;
//int si;
//const int si=14;
//const int si=15;

cl_device_id *devices;
cl_mem buffer[MAX];
cl_platform_id platform;
cl_uint num_devices;
cl_context context[MAX];
cl_program program[MAX];
cl_kernel kernel[MAX];
cl_command_queue cmd_queue[MAX];
cl_command_queue_properties properties;

//int spread;
//long lGTotal;
//long lGTotal;
uint64_t lGTotal;
//long lGUnique;
uint64_t lGUnique;
typedef int64_t qint;
//typedef int64_t qint;
enum { Place,Remove,Done };
struct queenState {
  //int BOUND1;
//  qint BOUND1;
  //int BOUND2;
  qint BOUND2;
  //int BOUND3;
  qint BOUND3;
//  int si;
  int id;
  //int aB[MAX];
  qint aB[MAX];
  long lTotal; // Number of solutinos found so far.
  //int step;
  char step;
  //int y;
  char y;
  //int startCol; // First column this individual computation was tasked with filling.
  char startCol; // First column this individual computation was tasked with filling.
  //int bm;
  qint bm;
  //long down;
  qint down;
  //long right;
  qint right;
  qint left;
} __attribute__((packed));

struct queenState inProgress[MAX*MAX];
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
  cl_int status; char value[BUFFER_SIZE]; size_t size;
  status=clGetPlatformIDs(1,&platform,NULL);
  if(status!=CL_SUCCESS){ printf("Couldn't get platform ID."); return 1; }
	status=clGetPlatformInfo(platform,CL_PLATFORM_NAME,BUFFER_SIZE,value,&size);
  if(status!=CL_SUCCESS){ printf("Couldn't get platform info."); return 2; }
  else{ if(DEBUG>0) printf("CL_PLATFORM_NAME:%s\n",value); }
	status=clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,value,&size);	
  if(status!=CL_SUCCESS){ printf("Couldn't get platform info."); return 3; }
  else{ if(DEBUG>0) printf("CL_PLATFORM_VERSION:%s\n",value); }
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
  cl_int status; char value[BUFFER_SIZE]; size_t size;
 	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&num_devices);
	if(status==CL_DEVICE_NOT_FOUND){
 		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&num_devices);
    if(DEBUG>0) printf("CL_DEVICE_TYPE_ALL\n");
	}
  if(status!=CL_SUCCESS){ printf("Couldn't get device count."); return 4; }
  else{ if(DEBUG>0) printf("CL_DEVICE COUNT:%d\n",num_devices); }
  devices=malloc(num_devices * sizeof(cl_device_id));
	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,num_devices,devices,NULL);
	if(status==CL_DEVICE_NOT_FOUND){
		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,num_devices,devices,NULL);
    if(DEBUG>0) printf("CL_DEVICE_TYPE_ALL\n");
	}else{ if(DEBUG>0) printf("CL_DEVICE_TYPE_GPU\n"); }
  if(status!=CL_SUCCESS){ printf("Couldn't get platform device count."); return 5; }
  else{ if(DEBUG>0) printf("CL_DEVICE INFO\n"); }
	for(int didx=0;didx<(int)num_devices;didx++){
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_NAME,BUFFER_SIZE,value,&size);	
    if(status!=CL_SUCCESS){ printf("Couldn't get device name."); return 6; }
    else{ if(DEBUG>0) printf(" +CL_DEVICE_NAME:%s\n",value); }
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_VERSION,BUFFER_SIZE,value,&size);
    if(status!=CL_SUCCESS){ printf("Couldn's get device version."); return 7; }
    else{ if(DEBUG>0) printf("  CL_DEVICE_VERSION:%s\n",value); }
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
int createContext(int BOUND1){
  cl_int status;
  context[BOUND1]=clCreateContext(NULL,num_devices,devices,NULL,NULL,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating context.\n"); return 8; }
	else{ if(DEBUG>0) printf("Creating context.\n"); }
  return 0;
}
/**
 * カーネルコードの読み込み
 */
void get_queens_code(char **buffer,int si,int BOUND1){
  char prefix[256];
  //int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n #define SIZE %d\n",si);
  //int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n//#define SIZE %d\n",si);
  //  int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n #define SIZE %d\n",si);
  int prefixLength=snprintf(prefix,256,"#define OPENCL_STYLE\n#define SIZE %d\n#define BOUND1 %d\n",si,BOUND1);
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
 * カーネルソースの読み込み
 */
int createProgramWithSource(int si,int BOUND1){
  cl_int status; char *code;
  get_queens_code(&code,si,BOUND1); //カーネルソースの読み込み
  if(code==NULL){ printf("Couldn't load the code.\n"); return 9; }
	else{ if(DEBUG>0) printf("Loading kernel code.\n"); }
  program[BOUND1]=clCreateProgramWithSource(context[BOUND1],1,(const char **) &code,NULL,&status);
  free(code);
  if(status!=CL_SUCCESS){ printf("Couldn't creating program."); return 10; }
	else{ if(DEBUG>0) printf("Creating program.\n"); }
  return 0;
}
/**
  プログラムのビルドとロギング
  clBuildProgram()カーネルオブジェクトを作成する。
  program    実行ファイルを作成するもとになるプログラム
  kernel_name    __kernelで事前に指定する関数名。
  kernel_name    __kernelで事前に指定する関数名。
  errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
*/
int buildProgram(int BOUND1){
  cl_int status;
  //status=clBuildProgram(program,num_devices,devices,NULL,NULL,NULL);
  status=clBuildProgram(program[BOUND1],num_devices,devices,NULL,NULL,NULL);
  if(status!=CL_SUCCESS){
    char log[2048];
   	status=clGetProgramBuildInfo(program[BOUND1],devices[0],CL_PROGRAM_BUILD_LOG,2048,log,NULL);
    if(DEBUG>0) printf("##### ERROR MESSAGE \n%s\n#####\n",log);
    return 11;
  }else{
    if(DEBUG>0) printf("Building program.\n");
  }
  return status;
}
/**
 *
 */
int create(int si,int BOUND1){
  int rst;
  while (1){
    createProgramWithSource(si,BOUND1);  
    rst=buildProgram(BOUND1);    
    if(rst==0){ break; }
  }
  return 0;
}
/** * カーネルオブジェクトの宣言 */
int createKernel(int BOUND1){
  cl_int status;
  kernel[BOUND1]=clCreateKernel(program[BOUND1],FUNC,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating kernel."); return 12; }
	else{ if(DEBUG>0) printf("Creating kernel.\n"); }
  return 0;
}
/**
 * コマンドキューの生成
 * clCreateCommandQueue()指定したデバイスのコマンドキューを作成する。
 * context   コンテキスト。
 * device    第１引数のcontextに関連づけられたデバイス。
 * properties    コマンドキューに適用するプロパティのリスト。
 * errcode_ret    エラーコードを格納する変数。
*/
int commandQueue(int si,int BOUND1){
  cl_int status;
	//cmd_queue=clCreateCommandQueue(context,devices[0],0,&status);
  properties= CL_QUEUE_PROFILING_ENABLE;
	cmd_queue[BOUND1]=clCreateCommandQueue(context[BOUND1], devices[0], properties, &status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating command queue."); return 13; }
	else{ if(DEBUG>0) printf("Creating command queue.\n"); }
  return 0;
} 
/**
 * カーネルオブジェクトをキューに入れる
 * デバイスメモリを確保しつつデータをコピー
 * clCreateBuffer()バッファオブジェクトを作成する。
 * context バッファオブジェクトを作成するために必要なOpenCLコンテキスト。
 * flags    「バッファオブジェクトをどのようなメモリ領域に割り当てるか」「メモリ領域
 * をどのように使用するか」のような割り当てやusageに関する情報を指定するビットフィールド。
 * CL_MEM_READ_WRITE カーネルにメモリ領域へのRead/Writeを許可する設定。
 * CL_MEM_USE_HOST_PTR デバイスメモリ内でhost_ptrを指定することにより、OpsnCL処理に
 * バッファをキャッシュすることを許可する。
 * size    割り当てられたバッファメモリオブジェクトのバイトサイズ
 * host_ptr    アプリケーションにより既に割り当てられているバッファデータへのポインタ。
 * errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
*/
int ceil_int_div(int i, int div) {
    return (i + div - 1) / div;
}
int ceil_int(int i, int div) {
    return ceil_int_div(i, div) * div;
}
/**
void* aligned_malloc(size_t required_bytes, size_t alignment) {
	void* p1; 	// original block
	void** p2; 	// aligned block
	int offset = alignment - 1 + sizeof(void*);
	if ((p1 = (void*)malloc(required_bytes + offset)) == NULL) {
		 return NULL;
	}
	p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
	p2[-1] = p1;
	return p2;
}
 */
/**
 *
 */
int makeInProgress(int si,int BOUND1){
  cl_int status;
//  struct queenState inProgress[si*si*si];
//  for(int i=0;i<si;i++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
//        inProgress[BOUND1*si*si+j*si+k].BOUND1=i;
        inProgress[j*si+k].BOUND2=j;
        inProgress[j*si+k].BOUND3=k;
        //inProgress[i*si*si+j*si+k].si=si;
        inProgress[j*si+k].id=j*si+k;
        for (int m=0;m< si;m++){ inProgress[j*si+k].aB[m]=m;}
        inProgress[j*si+k].lTotal=0;
        inProgress[j*si+k].step=0;
        inProgress[j*si+k].y=0;
        inProgress[j*si+k].startCol =1;
        inProgress[j*si+k].bm= (1 << si) - 1;
        inProgress[j*si+k].down=0;
        inProgress[j*si+k].right=0;
        inProgress[j*si+k].left=0;
      }
    }
//  }
  if(DEBUG>0) printf("Starting computation of Q(%d)\n",si);
  /**
   *
   */
  //cl_uint optimizedSize=ceil_int(sizeof(inProgress), 64);
  //cl_int *inputA = (cl_int*)aligned_malloc(optimizedSize, 4096);
  //buffer=clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,optimizedSize,inputA,&status);
  buffer[BOUND1] = clCreateBuffer(context[BOUND1], CL_MEM_READ_WRITE, sizeof(inProgress), NULL, &status);
  clRetainMemObject(buffer[BOUND1]);
  if(DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't create buffer.\n"); return 14; }
  /**
   *
   */
	struct queenState *ptrMappedA;
	ptrMappedA = clEnqueueMapBuffer(
      cmd_queue[BOUND1],      //投入キュー
      buffer[BOUND1],         //対象のOpenCLバッファ
      CL_FALSE,       //終了までブロックするか -> しない
      CL_MAP_READ|CL_MAP_WRITE, //CPUが書き込むためにMapする 
                      //(読み込みならCL_MAP_READ) 
                      //(両方ならCL_MAP_READ | CL_MAP_WRITE)
      0,              //オフセット
      //optimizedSize,  //マップするサイズ
      //ceil_int(sizeof(inProgress), 64),
      ceil_int(sizeof(inProgress), 64),
      0,              //この関数が待機すべきeventの数
      NULL,           //この関数が待機すべき関数のリストへのポインタ
      NULL,           //この関数の返すevent
      &status);
  if(DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't enque write buffer command."); return 16; }
  /**
   *  メモリバッファへコピー
   */
  memcpy(ptrMappedA,inProgress,sizeof(inProgress));
  /**
  for(int i=0;i<si;i++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
        ptrMappedA[i*si*si+j*si+k]=inProgress[i*si*si+j*si+k];
      }
    }
  }
  */
  /**
   * マップオブジェクトの解放
   */
  status = clEnqueueUnmapMemObject(
        cmd_queue[BOUND1],  //投入キュー
        buffer[BOUND1],     //対象のOpenCLバッファ
        ptrMappedA, //取得したホスト側のポインタ
        0,          //この関数が待機すべきeventの数
        NULL,       //この関数が待機すべき関数のリストへのポインタ
        NULL);      //この関数の返すevent
  if(DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't finish command queue."); return 14; }
 /**
    カーネルの引数をセット
    clSetKernelArg()カーネルの特定の引数に値をセットする。
    kernel    値をセットするカーネル。
    arg_index    引数のインデックス。
    arg_size    引数として渡すのデータのサイズ。
    arg_value    第２引数arg_indexで指定した引数にわたすデータへのポインタ。
  */
  status=clSetKernelArg(kernel[BOUND1],0,sizeof(cl_mem),&buffer[BOUND1]);
  if(DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't set kernel arg."); return 15; }
  return 0;
}
/**
 * タスクの終了を待機する
 */
//int all_tasks_done(struct queenState *tasks, int32_t num_tasks) {
int all_tasks_done(int32_t num_tasks) {
	for (int i = 0; i < num_tasks; i++)
		//if (tasks[i].step != 2)
		if (inProgress[i].step != 2)
			return 0;
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
int execKernel(int si,int BOUND1){
  cl_int status;
  while(!all_tasks_done(si*si)){
    size_t dim=1;
    size_t globalWorkSize[] = {si*si};
    size_t localWorkSize[] = { si };
    status=clEnqueueNDRangeKernel(
        cmd_queue[BOUND1],         //タスクを投入するキュー
        kernel[BOUND1],            //実行するカーネル
        dim,               //work sizeの次元
        NULL,              //NULLを指定すること
        globalWorkSize,    //全スレッド数
        localWorkSize,     //1グループのスレッド数
        0,                 //この関数が待機すべきeventの数
        NULL,              //この関数が待機すべき関数のリストへのポインタ
        NULL);             //この関数の返すevent
    if(DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't enque kernel execution command."); return 17; }
    /**
     * 結果を読み込み
     */
    status=clEnqueueReadBuffer(cmd_queue[BOUND1],buffer[BOUND1],CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
    if(DEBUG>0) if(status!=CL_SUCCESS){ printf("Couldn't enque read command."); return 18; }
  } //end while
  return 0;
}
/**
 * 結果の印字
 *
 */
int execPrint(int si,int BOUND1){
//  for(int BOUND1=0;BOUND1<si;BOUND1++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
          if(DEBUG>0) printf("%d: %ld\n",inProgress[j*si+k].id,inProgress[j*si+k].lTotal);
          lGTotal+=inProgress[j*si+k].lTotal;
        }
      }
//    }
  return 0;
}
/**
 * clGetProgramBuildInfo();		// プログラムのビルド情報を取得
 *
*/
int NQueens(int si){
  struct timeval t0;
  struct timeval t1;
  int ss;int ms;int dd;
  getPlatform();              // プラットフォーム一覧を取得
  getDeviceID();              // デバイス一覧を取得

  gettimeofday(&t0,NULL);    // 計測開始
  lGTotal=0;
  lGUnique=0;
  for(int i=0;i<si;i++){
    createContext(i);            // コンテキストの作成
    create(si,i);
    createKernel(i);             // カーネルの作成
    commandQueue(si,i);             // コマンドキュー作成
    //struct queenState inProgress[si*si*si];
    makeInProgress(si,i);
    execKernel(si,i);
    execPrint(si,i);
		clReleaseContext(context[i]);
		clReleaseProgram(program[i]);
		clReleaseKernel(kernel[i]);
		clReleaseCommandQueue(cmd_queue[i]);
  }
  gettimeofday(&t1,NULL);    // 計測終了
  free(devices);
	clReleaseMemObject(buffer);

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
  //printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", si,lGTotal,lGUnique,dd,hh,mm,ss,ms);
  printf("%2d:%18llu%18llu%12.2d:%02d:%02d:%02d.%02d\n", si,lGTotal,lGUnique,dd,hh,mm,ss,ms);
  return 0;
}
/**
 *
 *
 */
int main(void){
  int min=4;
  int targetN=22;
  printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=targetN;i++){
    lGTotal=0; lGUnique=0;
    NQueens(i);
  }
  return 0;
}
