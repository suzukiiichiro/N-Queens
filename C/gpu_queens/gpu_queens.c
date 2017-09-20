#include "stdio.h"
#include "string.h"
#ifdef __APPLE__ // MacOSであるかを確認
#include "OpenCL/cl.h" //MacOSの場合はインクルード
#else
#include <CL/cl.h> //Windows/Unix/Linuxの場合はインクルード
#endif
#define PROGRAM_FILE "./queen_kernel.c" //カーネルソースコード
#define FUNC "place" //カーネル関数の名称を設定

const int32_t si=13;
//const int32_t spread=1024;
const int32_t spread=si;
typedef int64_t qint;
enum { Place,Remove,Done };
struct queenState {
	int BOUND1;
  int id;
  qint aB[si];
  uint64_t lTotal;
  char step;
  char y;
  char startCol;
  qint bm;
  qint down;
  qint right;
  qint left;
} __attribute__((packed));
/**
 * カーネルコードの読み込み
 */
void get_queens_code(char ** buffer){
  char prefix[256];
  int prefixLength =
    snprintf(prefix,256,"#define OPENCL_STYLE\n#define si %d\n",si);
  FILE * f=fopen(PROGRAM_FILE,"rb");
  if(!f){
    *buffer=NULL;
    return;
  }
  long fileLength=0;
  fseek(f,0,SEEK_END);
  fileLength=ftell(f);
  fseek(f,0,SEEK_SET);
  long totalLength=prefixLength + fileLength + 1;
  *buffer=malloc(totalLength);
  strcpy(*buffer,prefix);
  if(buffer)
    fread(*buffer + prefixLength,1,fileLength,f);
  fclose(f);
  // Replace BOM with space
 (*buffer)[prefixLength]=' ';
 (*buffer)[prefixLength + 1]=' ';
 (*buffer)[prefixLength + 2]=' ';
}
int all_tasks_done(struct queenState * tasks,size_t num_tasks){
  for(int i=0; i <(int)num_tasks; i++)
    if(tasks[i].step==Done)
      return 1;
  return 0;
}
/**
main()OpenCL 主な流れ 
clGetPlatformIDs();         // プラットフォーム一覧を取得
clGetDeviceIDs();           // デバイス一覧を取得
clCreateContext();          // コンテキストの作成
clCreateProgramWithSource();// ソースコードからカーネルプログラム作成
clBuildProgram();						// カーネルプログラムのビルド
clGetProgramBuildInfo();		// プログラムのビルド情報を取得
clCreateKernel();						// カーネルの作成
clCreateCommandQueue();     // コマンドキュー作成
clCreateBuffer();           // 書き込み・読み込みメモリバッファの作成
clSetKernelArg();           // カーネル引数の設定
clEnqueueWriteBuffer();     // メモリバッファへの書き込み
clEnqueueNDRangeKernel();   // カーネル実行
clEnqueueReadBuffer();      // メモリバッファから結果読み出し
clFinish();                 // 実行が終わるまで待機
// それぞれの終了処理
clReleaseMemObject();
clReleaseKernel();
clReleaseProgram();
clReleaseCommandQueue();
clReleaseContext();
*/
#define BUFFER_SIZE 4096
int main(){
  /* OpneCL APIの変数 */
  cl_int status;

	char value[BUFFER_SIZE];
	size_t size;
	cl_event profile_event;
  /**
	プラットフォーム一覧を取得
	clGetPlatformIDs()使用できるプラットフォームの数とID一覧を取得する関数
	戻り値　CL_SUCCESS 成功 CL_INVALID_VALUE 失敗
	PLATFORM : 見つかったプラットフォームの最大取得数
	platforms : 見つかったプラットフォームの一覧が代入されるポインタ
	&platformCount : 使用できるプラットフォームの数が代入されるポインタ  
  */

  /** プラットフォームオブジェクトの宣言 */
  cl_platform_id platform;
  /** OpenCLデバイスのプラットフォームの特定*/
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
    printf("CL_PLATFORM_NAME:%s\n",value);
  }
	status=clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,value,&size);	
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform info.");
    return 3; 
  }else{
    printf("CL_PLATFORM_VERSION:%s\n",value);
  }
	// デバイス一覧を取得
	// clGetDeviceIds()使用できるデバイスの数とID一覧を取得する関数
	// platforms[0]clGetPlatformIDs関数で取得したプラットフォームのID
	// CL_DEVICE_TYPE_CPU：ホストプロセッサを指定する
	// CL_DEVICE_TYPE_GPU：GPUデバイスを指定する。
	// CL_DEVICE_TYPE_ACCELERATOR：OpenCL専用デバイスを指定する
	// CL_DEVICE_TYPE_CUSTOM：OpenCL C言語で実装されたプログラムに対応していないデバイスを指定する。
	// CL_DEVICE_TYPE_DEFAULT：システム上で設定されているデフォルトのOpenCLデバイス。
	// この指定はCL_DEVICE_TYPE_CUSTOMと併用できない。
	// CL_DEVICE_TYPE_ALL：C言語で実装されたプログラムに対応していないデバイスを除いたす
	// べての使用可能なOpenCLデバイスを指定する。
	// DEVICE_MAX : 取得するデバイスの制限数。
	// devices : 見つかったOpenCLデバイスID一覧を取得するためのポインタ。
	// &deviceCount : 第３引数device_typeに適合したOpenCLデバイスの数を取得するためのポインタ。
	//デバイスのカウント
  cl_uint num_devices;
 	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&num_devices);
	if(status==CL_DEVICE_NOT_FOUND){
 		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&num_devices);
		printf("CL_DEVICE_TYPE_ALL\n");
	}
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get device count.");
    return 4; 
  }else{
	  printf("CL_DEVICE COUNT:%d\n",num_devices);
  }
	//デバイスIDの取得

  /** デバイスオブジェクトの宣言 */
 	cl_device_id * devices=malloc(num_devices * sizeof(cl_device_id));
	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,num_devices,devices,NULL);
	if(status==CL_DEVICE_NOT_FOUND){
		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,num_devices,devices,NULL);
		printf("CL_DEVICE_TYPE_ALL\n");
	}else{
		printf("CL_DEVICE_TYPE_GPU\n");
	}
  if(status!=CL_SUCCESS){ 
    printf("Couldn't get platform device count.");
    return 5; 
  }else{
    printf("CL_DEVICE INFO\n");
  }
	for(int didx=0;didx<(int)num_devices;didx++){
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_NAME,BUFFER_SIZE,value,&size);	
    if(status!=CL_SUCCESS){
      printf("Couldn't get device name.");
      return 6;
    }else{
      printf(" +CL_DEVICE_NAME:%s\n",value);
    }
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_VERSION,BUFFER_SIZE,value,&size);
    if(status!=CL_SUCCESS){
      printf("Couldn's get device version.");
      return 7;
    }else{
      printf("  CL_DEVICE_VERSION:%s\n",value);
    }
	}
	// コンテキストの作成
	// clCreateContext()ひとつ以上のデバイスで使用するためのコンテキストを作成する。
	// nullptr コンテキストプロパティを指定する。
	// 各プロパティ名にはそれぞれに対応した要求される値が続く。この一覧の終端には0がつけ
	// られる。引数porpertiesには、処理依存のプラットフォームの場合に限りNULLを指定する
	// ことができる。
	// 1 : 第３引数devicesで指定されたデバイスの数
	// devices : 一意に定まる、clGetDeviceIDs関数で取得されたデバイス、また
	// はclCreateleftDevicesで作成されたサブデバイス。
	// nullptr : アプリケーションによって登録することが可能なコールバック関数。
	// nullptr : 引数pfn_notifyで設定したコールバック関数が呼び出されたとき、データが
	// 渡されるポインタ。この引数はNULLにした場合、無視される
	// &err エラーが発生した場合、そのエラーに合わせたエラーコードが返される。

  /** コンテキストオブジェクトの宣言 */
  cl_context context=clCreateContext(NULL,num_devices,devices,NULL,NULL,&status);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating context.\n");
    return 8; 
  }else{
    printf("Creating context.\n");
  }
  //カーネルコードの読み込み
  char * code;
  get_queens_code(&code);
  if(code==NULL){
    printf("Couldn't load the code.");
    return 9;
  }else{
    printf("Loading kernel code.\n");
  }
	/** プログラムオブジェクトの宣言 */
  cl_program program=clCreateProgramWithSource(context,1,(const char **)&code,NULL,&status);
  free(code);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating program.");
    return 10; 
  }else{
    printf("Creating program.\n");
  }
	// プログラムのビルド
	// clBuildProgram()カーネルオブジェクトを作成する。
	// program    実行ファイルを作成するもとになるプログラム
	// kernel_name    __kernelで事前に指定する関数名。
	// errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
  status=clBuildProgram(program,num_devices,devices,NULL,NULL,NULL);
  if(status!=CL_SUCCESS){
    char log[2048];
   	status=clGetProgramBuildInfo(program,devices[0],CL_PROGRAM_BUILD_LOG,2048,log,NULL);
    printf("%s",log);
    printf("Couldn't building program.");
    return 11;
  }else{
    printf("Building program.\n");
  }
	/** カーネルオブジェクトの宣言 */
  cl_kernel kernel=clCreateKernel(program,FUNC,&status);
  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating kernel.");
    return 12; 
  }else{
    printf("Creating kernel.\n");
  }
	// コマンドキューの作成
	// clCreateCommandQueue()指定したデバイスのコマンドキューを作成する。
	// context    OpenCLコンテキスト。
	// device    第１引数のcontextに関連づけられたデバイス。
	// properties    コマンドキューに適用するプロパティのリスト。
	// errcode_ret    エラーコードを格納する変数。

  /** コマンドキューオブジェクトの宣言 */
 	cl_command_queue cmd_queue=clCreateCommandQueue(context,devices[0],0,&status);
	/*
 	cl_command_queue cmd_queue[num_devices];
  for(int i=0;i<num_devices;i++){
    cmd_queue[i]=clCreateCommandQueue(context,devices[i],0,&status);
  }
	*/

  if(status!=CL_SUCCESS){ 
    printf("Couldn't creating command queue.");
    return 13; 
  }else{
    printf("Creating command queue.\n");
  }
  /**
   * 初期化
   */
  // List of in-progress tasks
  struct queenState inProgress[spread]={ 0 };
  for(int i=0; i < spread; i++){
    struct queenState s={ 0 };
		s.BOUND1=i; //BOUND1の初期化
    s.id=i;
    s.bm=(1<<si)-1;
    inProgress[i]=s;
		for (int i=0; i < si; i++){
			s.aB[i]=i;
		}
		s.lTotal=0;
		s.step=0;
		s.y=0;
		s.startCol =1;
		s.down=0;
		s.right=0;
		s.left=0;
  }
	// デバイスメモリを確保しつつデータをコピー
	// clCreateBuffer()バッファオブジェクトを作成する。
	// context バッファオブジェクトを作成するために必要なOpenCLコンテキスト。
	// flags    「バッファオブジェクトをどのようなメモリ領域に割り当てるか」「メモリ領域
	// をどのように使用するか」のような割り当てやusageに関する情報を指定するビットフィールド。
	// CL_MEM_READ_WRITE カーネルにメモリ領域へのRead/Writeを許可する設定。
	// CL_MEM_USE_HOST_PTR デバイスメモリ内でhost_ptrを指定することにより、OpsnCL処理に
	// バッファをキャッシュすることを許可する。
	// size    割り当てられたバッファメモリオブジェクトのバイトサイズ
	// host_ptr    アプリケーションにより既に割り当てられているバッファデータへのポインタ。
	// errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
  printf("Starting computation of Q(%d)\n",si);
  while(!all_tasks_done(inProgress,spread)){
    printf("loop\n");
    cl_mem buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(inProgress),NULL,&status);
    if(status!=CL_SUCCESS){
      printf("Couldn't create buffer.\n");
      return 14;
    }
    // カーネルの引数をセット
    // clSetKernelArg()カーネルの特定の引数に値をセットする。
    // kernel    値をセットするカーネル。
    // arg_index    引数のインデックス。
    // arg_size    引数として渡すのデータのサイズ。
    // arg_value    第２引数arg_indexで指定した引数にわたすデータへのポインタ。
    status=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
    if(status!=CL_SUCCESS){
      printf("Couldn't set kernel arg.");
      return 15;
    }
		//メモリバッファへの書き込み
    status=clEnqueueWriteBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
		/**
    for(int i=0;i<num_devices;i++){
      status=clEnqueueWriteBuffer(cmd_queue[i],buffer,CL_TRUE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
    }
		*/
    if(status!=CL_SUCCESS){
      printf("Couldn't enque write buffer command.");
      return 16;
    }
		//カーネルの実行
    size_t globalSizes[]={ spread };
   	status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,&profile_event);
		/**
    for(int i=0;i<num_devices;i++){
   	  status=clEnqueueNDRangeKernel(cmd_queue[i],kernel,1,0,globalSizes,NULL,0,NULL,&profile_event);
    }
		*/
    if(status!=CL_SUCCESS){
      printf("Couldn't enque kernel execution command.");
      return 17;
    }
		//結果を読み込み
   	status=clEnqueueReadBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
		/**
    for(int i=0;i<num_devices;i++){
   	  status=clEnqueueReadBuffer(cmd_queue[i],buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
    }
		*/
    if(status!=CL_SUCCESS){
      printf("Couldn't enque read command.");
      return 18;
    }
		//実行が終わるまで待機
    status=clFinish(cmd_queue);
		/**
    for(int i=0;i<num_devices;i++){
      status=clFinish(cmd_queue[i]);
    }
		*/
    if(status!=CL_SUCCESS){
      printf("Couldn't finish command queue.");
      return 19;
    }
  }//end while

	//結果の印字
  long lGTotal=0;
  for(int i=0; i < spread; i++){
    printf("%d: %llu\n",inProgress[i].id,inProgress[i].lTotal);
    lGTotal+=inProgress[i].lTotal;
	}
  printf("lGTotal:%ld\n",lGTotal);

	cl_ulong ev_start_time=(cl_ulong)0;
	cl_ulong ev_end_time=(cl_ulong)0;
	double execution_time=0.0;
	status=clGetEventProfilingInfo(profile_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&ev_start_time,NULL);
	status=clGetEventProfilingInfo(profile_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&ev_end_time,NULL);
	execution_time=ev_end_time-ev_start_time;
  printf("Exe time in seconds: %0.3e\n",execution_time/1000000000);


  free(devices);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  //clReleaseCommandQueue(cmd_queue[0]);
  clReleaseCommandQueue(cmd_queue);
  clReleaseContext(context);
  return 0;
}
