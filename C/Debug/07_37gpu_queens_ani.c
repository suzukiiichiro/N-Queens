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
13:             73712                 0          00:00:00:00.06
14:            365596                 0          00:00:00:00.28
15:           2279184                 0          00:00:00:01.37
16:          14772512                 0          00:00:00:08.19
17:          95815104                 0          00:00:00:55.14
18:         666090624                 0          00:00:06:31.63
19:        4968057848                 0          00:00:48:07.87
 
*/

#include "stdio.h"
#include "string.h"
#ifdef __APPLE__ // MacOSであるかを確認
#include "OpenCL/cl.h" //MacOSの場合はインクルード
#else
#include<CL/cl.h> //Windows/Unix/Linuxの場合はインクルード
#endif

#define PROGRAM_FILE "./07_37queen_kernel.c" //カーネルソースコード
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
  int BOUND1;
  int BOUND2;
  int BOUND3;
  int si;
  int id;
  int aB[MAX];
  long lTotal; // Number of solutinos found so far.
  int step;
  int y;
  int startCol; // First column this individual computation was tasked with filling.
  int bm;
  long down;
  long right;
  long left;
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
    printf("Couldn't load the code.\n");
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
  //status=clBuildProgram(program,num_devices,devices,NULL,NULL,NULL);
  status=clBuildProgram(program,1,devices,"",NULL,NULL);
  if(status!=CL_SUCCESS){
    char log[2048];
   	status=clGetProgramBuildInfo(program,devices[0],CL_PROGRAM_BUILD_LOG,2048,log,0);
		/***/
    if(DEBUG>0) printf("##### ERROR MESSAGE \n%s\n#####\n",log);
		/***/
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
	//cmd_queue=clCreateCommandQueue(context,devices[0],0,&status);
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cmd_queue = clCreateCommandQueue(context, devices[0], properties, &status);
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
int ceil_int_div(int i, int div) {
    return (i + div - 1) / div;
}
int ceil_int(int i, int div) {
    return ceil_int_div(i, div) * div;
}
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
int makeInProgress(int si){
  cl_int status;
	cl_int* inputA ;

  struct queenState inProgress[si*si*si];
//  struct queenState s;
	struct queenState *ptrMappedA ;

  for(int i=0;i<si;i++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
        inProgress[i*si*si+j*si+k].BOUND1=i;
        inProgress[i*si*si+j*si+k].BOUND2=j;
        inProgress[i*si*si+j*si+k].BOUND3=k;
        inProgress[i*si*si+j*si+k].si=si;
        inProgress[i*si*si+j*si+k].id=i*si*si+j*si+k;
        for (int m=0;m< si;m++){ inProgress[i*si*si+j*si+k].aB[m]=m;}
        inProgress[i*si*si+j*si+k].lTotal=0;
        inProgress[i*si*si+j*si+k].step=0;
        inProgress[i*si*si+j*si+k].y=0;
        inProgress[i*si*si+j*si+k].startCol =1;
        inProgress[i*si*si+j*si+k].bm= (1 << si) - 1;
        inProgress[i*si*si+j*si+k].down=0;
        inProgress[i*si*si+j*si+k].right=0;
        inProgress[i*si*si+j*si+k].left=0;
        //inProgress[i*si*si+j*si+k]=s;
      }
    }
  }
  if(DEBUG>0) printf("Starting computation of Q(%d)\n",si);
	cl_uint optimizedSize=ceil_int(sizeof(inProgress), 64);

		inputA = (cl_int*)aligned_malloc(optimizedSize, 4096);
    //buffer=clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,optimizedSize,inputA,&status);
		buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(inProgress), NULL, &status);
    clRetainMemObject(buffer);
    if(status!=CL_SUCCESS){ printf("Couldn't create buffer.\n"); return 14; }
		ptrMappedA = clEnqueueMapBuffer(
				cmd_queue, //投入キュー
				buffer,         //対象のOpenCLバッファ
				CL_FALSE,         //終了までブロックするか -> しない
				CL_MAP_READ|CL_MAP_WRITE,     //CPUが書き込むためにMapする 
																			//(読み込みならCL_MAP_READ) 
																			//(両方ならCL_MAP_READ | CL_MAP_WRITE)
				0,                //オフセット
				//sizeof(inProgress)*si*si, //マップするサイズ
				optimizedSize, //マップするサイズ
				0,      //この関数が待機すべきeventの数
				NULL,   //この関数が待機すべき関数のリストへのポインタ
				NULL,   //この関数の返すevent
				&status);
//		status=clFinish(cmd_queue);
    if(status!=CL_SUCCESS){ printf("Couldn't enque write buffer command."); return 16; }

		//メモリバッファへコピー
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
		//マップオブジェクトの解放
		status = clEnqueueUnmapMemObject(
					cmd_queue, //投入キュー
					buffer,//対象のOpenCLバッファ
					ptrMappedA, //取得したホスト側のポインタ
					0, //この関数が待機すべきeventの数
					NULL, //この関数が待機すべき関数のリストへのポインタ
					NULL);//この関数の返すevent
		//終了を待機
//	  status = clFinish(cmd_queue);
    if(status!=CL_SUCCESS){ printf("Couldn't finish command queue."); return 14; }

    status=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
    if(status!=CL_SUCCESS){ printf("Couldn't set kernel arg."); return 15; }
  while(!all_tasks_done(inProgress,si*si*si)){
    //printf("loop\n");
		/** 
			メモリバッファへの書き込み 
		  対応するホスト側のポインタを取得する
		*/
    //status=clEnqueueWriteBuffer(cmd_queue,buffer,CL_FALSE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
		/**
			カーネルの引数をセット
			clSetKernelArg()カーネルの特定の引数に値をセットする。
			kernel    値をセットするカーネル。
			arg_index    引数のインデックス。
			arg_size    引数として渡すのデータのサイズ。
			arg_value    第２引数arg_indexで指定した引数にわたすデータへのポインタ。
		*/
		/**
			カーネルの実行 カーネルを実行するコマンドをキューに入れて、カーネル関数をデバイスで実行
			work sizeの指定
			ここでは1要素に対して1 work item
			またグループあたり1 work item (実は効率的でない)
			width * heightの2次元でwork itemを作成
		*/
		size_t dim=1;
		//size_t globalWorkSize[] = {si*si*si*si};
		size_t globalWorkSize[] = {si*si*si};
		//size_t globalWorkSize[] = {si*si*si,0,0};
		size_t localWorkSize[] = { si };
		//size_t localWorkSize[] = {1};
		//size_t localWorkSize[] = { CL_DEVICE_MAX_WORK_ITEM_SIZES };
		//タスクをキューに積む
		status=clEnqueueNDRangeKernel(
				cmd_queue, //タスクを投入するキュー
				kernel,       //実行するカーネル
				dim,               //work sizeの次元
				NULL,              //NULLを指定すること
				globalWorkSize,    //全スレッド数
				localWorkSize,     //1グループのスレッド数
				0,                 //この関数が待機すべきeventの数
				NULL,              //この関数が待機すべき関数のリストへのポインタ
				NULL);             //この関数の返すevent
//    status=clFinish(cmd_queue);
    if(status!=CL_SUCCESS){ printf("Couldn't enque kernel execution command."); return 17; }
		//結果を読み込み
   	status=clEnqueueReadBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
//		status=clFinish(cmd_queue);
    if(status!=CL_SUCCESS){ printf("Couldn't enque read command."); return 18; }
  }//end while
	//結果の印字
  lGTotal=0;
  lGUnique=0;
  for(int i=0;i<si;i++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
          //for(int i=0;i<SPREAD;i++){
          if(DEBUG>0) printf("%d: %ld\n",inProgress[i*si*si+j*si+k].id,inProgress[i*si*si+j*si+k].lTotal);
          lGTotal+=inProgress[i*si*si+j*si+k].lTotal;
        }
      }
    }
  /*
  for(int i=0;i<si;i++){
    if(DEBUG>0) printf("%d: %ld\n",inProgress[i].id,inProgress[i].lTotal);
    lGTotal+=inProgress[i].lTotal;
	}
  */
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
  while (1){
    createProgramWithSource();  // ソースコードからカーネルプログラム作成
    int rst=buildProgram();             // カーネルプログラムのビルド
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
  int min=10;
  int targetN=22;
  printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  //for(int i=min;i<=MAX;i++){
  for(int i=min;i<=targetN;i++){
  //for(int i=targetN;i<=targetN;i++){
    lGTotal=0; lGUnique=0;
    NQueens(i);
  }
  return 0;
}
