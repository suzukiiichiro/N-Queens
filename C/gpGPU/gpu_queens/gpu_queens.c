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
#define DEBUG 0
//#define SPREAD 1024
//
//const int32_t si=13;
//const int si=13;
//int si;
//const int si=14;
//const int si=15;

cl_platform_id platform;
cl_uint num_devices;
cl_device_id *devices;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue cmd_queue;
cl_mem buffer;

int spread;
long lGTotal;
long lGUnique;
//typedef int64_t qint;
enum { Place,Remove,Done };
struct queenState {
  int si;
  int id;
  int aB[MAX];
  long lTotal; // Number of solutinos found so far.
  int step;
  int y;
  int startCol; // First column this individual computation was tasked with filling.
  int bm;
  int down;
  int right;
  int left;
} __attribute__((packed));

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
 * タスクの終了を待機する
 */
//int all_tasks_done(struct queenState * tasks,size_t num_tasks){
int all_tasks_done(struct queenState * tasks, int32_t num_tasks) {
	for (int i = 0; i < num_tasks; i++)
		if (tasks[i].step != 2)
			return 0;
	return 1;
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
  status=clGetPlatformIDs(1,&platform,NULL);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform ID.");
    return 1; 
  }
	status=clGetPlatformInfo(platform,CL_PLATFORM_NAME,BUFFER_SIZE,value,&size);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform info.");
    return 2; 
  }else{
    if(DEBUG>0) printf("CL_PLATFORM_NAME:%s\n",value);
  }
	status=clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,value,&size);	
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform info.");
    return 3; 
  }else{
    if(DEBUG>0) printf("CL_PLATFORM_VERSION:%s\n",value);
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
    if(DEBUG>0) printf("CL_DEVICE_TYPE_ALL\n");
	}
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get device count.");
    return 4; 
  }else{
    if(DEBUG>0) printf("CL_DEVICE COUNT:%d\n",num_devices);
  }
  //cl_device_id * devices=malloc(num_devices * sizeof(cl_device_id));
  devices=malloc(num_devices * sizeof(cl_device_id));
	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,num_devices,devices,NULL);
	if(status==CL_DEVICE_NOT_FOUND){
		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,num_devices,devices,NULL);
    if(DEBUG>0) printf("CL_DEVICE_TYPE_ALL\n");
	}else{
    if(DEBUG>0) printf("CL_DEVICE_TYPE_GPU\n");
	}
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform device count.");
    return 5; 
  }else{
    if(DEBUG>0) printf("CL_DEVICE INFO\n");
  }
	for(int didx=0;didx<(int)num_devices;didx++){
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_NAME,BUFFER_SIZE,value,&size);	
    if(status!=CL_SUCCESS){
      printf("Couldn't get device name.");
      return 6;
    }else{
      if(DEBUG>0) printf(" +CL_DEVICE_NAME:%s\n",value);
    }
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_VERSION,BUFFER_SIZE,value,&size);
    if(status!=CL_SUCCESS){
      printf("Couldn's get device version.");
      return 7;
    }else{
      if(DEBUG>0) printf("  CL_DEVICE_VERSION:%s\n",value);
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
    if(DEBUG>0) printf("Creating context.\n");
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
    printf("Couldn't load the code.");
    return 9;
  }else{
    if(DEBUG>0) printf("Loading kernel code.\n");
  }
  program=clCreateProgramWithSource(context,1,(const char **)&code,NULL,&status);
  free(code);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating program.");
    return 10; 
  }else{
    if(DEBUG>0) printf("Creating program.\n");
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
  status=clBuildProgram(program,num_devices,devices,NULL,NULL,NULL);
  if(status!=CL_SUCCESS){
    char log[2048];
   	status=clGetProgramBuildInfo(program,devices[0],CL_PROGRAM_BUILD_LOG,2048,log,NULL);
    //printf("%s",log);
    if(DEBUG>0) printf("Couldn't building program.");
    return 11;
  }else{
    if(DEBUG>0) printf("Building program.\n");
  }
  return status;
}
/** カーネルオブジェクトの宣言 */
int createKernel(){
  cl_int status;
  kernel=clCreateKernel(program,FUNC,&status);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating kernel.");
    return 12; 
  }else{
    if(DEBUG>0) printf("Creating kernel.\n");
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
	cmd_queue=clCreateCommandQueue(context,devices[0],0,&status);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating command queue.");
    return 13; 
  }else{
    if(DEBUG>0)
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
int makeInProgress(int si){
  cl_int status;
  struct queenState inProgress[si];
  //struct queenState inProgress[SIZE]={0};
  //struct queenState inProgress[SPREAD]={0};
  //struct queenState inProgress[SPREAD];
  for(int i=0;i<si;i++){
  //for(int i=0;i<SPREAD;i++){
    struct queenState s;
    s.si=si;
    s.id=i;
		for (int j=0;j< si;j++){ s.aB[j]=j;}
		s.lTotal=0;
		s.step=0;
    s.y=0;
		s.startCol =0;
    s.bm= (1 << si) - 1;
    s.down=0;
    s.right=0;
    s.left=0;
//
    int si=s.si;
    int id=s.id;
    int aB[MAX];
    for (int j = 0; j < si; j++)
    aB[j] = s.aB[j];

  uint64_t lTotal = s.lTotal;
  int step      = s.step;
  int y       = s.y;
  int startCol  = s.startCol;
  int endCol  = 1;
  int bm     = s.bm;
  int down     = s.down;
  int right      = s.right;
  int left      = s.left;
  int BOUND1   = i;
  int msk = (1 << si) - 1;
  //printf("bound:%d:startCol:%d\n", BOUND1,startCol);
  uint16_t j = 1;
  while (j != 0)
  {
  	j++;
    if (y == endCol){
        step = 0;
        break;
    }
    if (step == Remove)
    {
      if (y == startCol)
      {
        step = Done;
        break;
      }
      --y;
      bm = aB[y];
    }
    int bit;
    if(y==0){
      if(bm & (1<<BOUND1)){
        bit=1<<BOUND1;
      }else{
        step=Done;
        break;
      }
    }else{
      bit = bm & -bm;
    }
    down ^= bit;
    right  ^= bit << y;
    left  ^= bit << (si - 1 - y);

    if (step == Place)
    {
      aB[y] = bm;
      ++y;

      if (y != si)
      {
        bm = msk & ~(down | (right >> y) | (left >> ((si - 1) - y)));

        if (bm == 0)
          step = Remove;
      }
      else
      {
        lTotal += 1;
        step = Remove;
      }
    }
    else
    {
      bm ^= bit;
      if (bm == 0)
        step = Remove;
      else
        step = Place;
    }
  }
  // Save kernel state for next round.
    s.si      =si;
    s.id      = id;
  for (int j = 0; j < si; j++)
    s.aB[j] = aB[j];
    s.lTotal = lTotal;
    s.step      = step;
    s.y       = y;
    s.startCol  = endCol;
    s.bm      = bm;
    s.down      = down;
    s.right       = right;
    s.left       = left;
   // printf("id:%d:ltotal:%ld:step:%d:y:%d:startCol:%d:bm:%d:down:%d:right:%d:left:%d:BOUND1:%d\n",s.id,s.lTotal,s.step,s.y,s.startCol,s.bm,s.down,s.right,s.left,s.BOUND1);
    inProgress[i]=s;
  }
  if(DEBUG>0) printf("Starting computation of Q(%d)\n",si);
  //while(!all_tasks_done(inProgress,SPREAD)){
  while(!all_tasks_done(inProgress,si)){
    //printf("loop\n");
    buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(inProgress),NULL,&status);
    if(status!=CL_SUCCESS){
      printf("Couldn't create buffer.\n");
      return 14;
    }
		/** メモリバッファへの書き込み */
    status=clEnqueueWriteBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
    if(status!=CL_SUCCESS){
      printf("Couldn't enque write buffer command.");
      return 16;
    }
		/**
    カーネルの引数をセット
    clSetKernelArg()カーネルの特定の引数に値をセットする。
    kernel    値をセットするカーネル。
    arg_index    引数のインデックス。
    arg_size    引数として渡すのデータのサイズ。
    arg_value    第２引数arg_indexで指定した引数にわたすデータへのポインタ。
		*/
    status=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
    if(status!=CL_SUCCESS){
      printf("Couldn't set kernel arg.");
      return 15;
    }
		//カーネルの実行 カーネルを実行するコマンドをキューに入れて、カーネル関数をデバイスで実行
    //size_t globalSizes[]={ spread*spread };
    //size_t globalSizes[]={ si*si };
    size_t globalSizes[]={ si };
    //printf("17\n");
   	status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
    if(status!=CL_SUCCESS){
      printf("Couldn't enque kernel execution command.");
      return 17;
    }
		//実行が終わるまで待機
    status=clFinish(cmd_queue);
    if(status!=CL_SUCCESS){ 
      printf("Couldn't finish command queue.");
      return 14;
    }
		//結果を読み込み
   	status=clEnqueueReadBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
    if(status!=CL_SUCCESS){
      printf("Couldn't enque read command.");
      return 18;
    }
  }//end while
	//結果の印字
  lGTotal=0;
  lGUnique=0;
  for(int i=0;i<si;i++){
  //for(int i=0;i<SPREAD;i++){
    if(DEBUG>0) printf("%d: %ld\n",inProgress[i].id,inProgress[i].lTotal);
    lGTotal+=inProgress[i].lTotal;
	}
  return 0;
}
int finalFree(){
  free(devices);
	clReleaseMemObject(buffer);
  clReleaseContext(context);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(cmd_queue);
  return 0;
}
int create(){
  //printf("create\n");
  while (1){
    createProgramWithSource();  // ソースコードからカーネルプログラム作成
    int rst=buildProgram();             // カーネルプログラムのビルド
    //printf("status:%d:::",rst);
    if(rst==0){
      break;
    }
  }
  return 0;
}
/**
	clGetProgramBuildInfo();		// プログラムのビルド情報を取得
*/
int NQueens(int si){
  struct timeval t0;
  struct timeval t1;
  int ss;int ms;int dd;
  getPlatform();              // プラットフォーム一覧を取得
  getDeviceID();              // デバイス一覧を取得
  createContext();            // コンテキストの作成
  create();
  createKernel();             // カーネルの作成
  commandQueue();             // コマンドキュー作成
  gettimeofday(&t0, NULL);    // 計測開始
  makeInProgress(si);
	// 1. clCreateBuffer();          // 書き込み・読み込みメモリバッファの作成
	// 2. clSetKernelArg();          // カーネル引数の設定
	// 3. clEnqueueWriteBuffer();    // メモリバッファへの書き込み
	// 4. clEnqueueNDRangeKernel();  // カーネル実行
	// 5. clEnqueueReadBuffer();     // メモリバッファから結果読み出し
	// 6. clFinish();                // 実行が終わるまで待機
  gettimeofday(&t1, NULL);      // 計測終了
  finalFree();                  // 解放
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
  printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", si,lGTotal,lGUnique,dd,hh,mm,ss,ms);
  return 0;
}
int main(void){
  int min=4;
  int targetN=19;
  printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  //for(int i=min;i<=MAX;i++){
  for(int i=min;i<=targetN;i++){
  //for(int i=targetN;i<=targetN;i++){
    lGTotal=0; lGUnique=0;
    NQueens(i);
  }
  return 0;
}
