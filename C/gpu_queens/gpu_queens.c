#include "stdio.h"
#include "string.h"
#include "OpenCL/cl.h"
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
  FILE * f=fopen("./queen_kernel.c","rb");
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
#define USE_GPU 0
#define BUFFER_SIZE 4096
int main(){
  cl_int status;
	char value[BUFFER_SIZE];
	size_t size;
	// プラットフォーム一覧を取得
	// clGetPlatformIDs()使用できるプラットフォームの数とID一覧を取得する関数
	// 戻り値　CL_SUCCESS 成功 CL_INVALID_VALUE 失敗
	// PLATFORM : 見つかったプラットフォームの最大取得数
	// platforms : 見つかったプラットフォームの一覧が代入されるポインタ
	// &platformCount : 使用できるプラットフォームの数が代入されるポインタ  
	//status=clGetPlatformIDs(1,&platform,NULL);
  cl_platform_id platform;
  status=clGetPlatformIDs(1,&platform,NULL);
  if(status!=CL_SUCCESS){
    return 1;
  }
	status=clGetPlatformInfo(platform,CL_PLATFORM_NAME,BUFFER_SIZE,value,&size);
	printf("CL_PLATFORM_NAME:%s\n",value);
	status=clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,value,&size);	
	printf("CL_PLATFORM_VERSION:%s\n",value);
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
  cl_uint num_devices;




	//デバイスのカウント
 	status=clGetDeviceIDs(platform,USE_GPU?CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_ALL,0,NULL,&num_devices);
	printf("CL_DEVICE COUNT:%d\n",num_devices);
  if(status!=CL_SUCCESS){
    return 2;
	}
	//デバイスIDの取得
 	cl_device_id * devices=malloc(num_devices * sizeof(cl_device_id));
	status=clGetDeviceIDs(platform,USE_GPU?CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_ALL,num_devices,devices,NULL);
	printf("CL_DEVICE INFO\n");
  if(status!=CL_SUCCESS){
    return 3;
	}
	for(int didx=0;didx<(int)num_devices;++didx){
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_NAME,BUFFER_SIZE,value,&size);	
		printf(" +CL_DEVICE_NAME:%s\n",value);
		status=clGetDeviceInfo(devices[didx],CL_DEVICE_VERSION,BUFFER_SIZE,value,&size);
		printf("  CL_DEVICE_VERSION:%s\n",value);
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
  printf("Creating context.\n");
  cl_context context=clCreateContext(NULL,num_devices,devices,NULL,NULL,&status);
  if(status!=CL_SUCCESS){
    return 4;
	}
  printf("Loading kernel code.\n");
  char * code;
  get_queens_code(&code);//カーネルコードの読み込み
  if(code==NULL){
    printf("Couldn't load the code.");
  }
	//コンテキストの作成
  printf("Creating program.\n");
  cl_program program=clCreateProgramWithSource(context,1,(const char **)&code,NULL,&status);
  free(code);
  if(status!=CL_SUCCESS){
    return 5;
	}
	// プログラムのビルド
	// clBuildProgram()カーネルオブジェクトを作成する。
	// program    実行ファイルを作成するもとになるプログラム
	// kernel_name    __kernelで事前に指定する関数名。
	// errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
  printf("Building program.\n");
  status=clBuildProgram(program,num_devices,devices,NULL,NULL,NULL);

  if(status!=CL_SUCCESS){
    char log[2048];
   	status=clGetProgramBuildInfo(program,devices[0],CL_PROGRAM_BUILD_LOG,2048,log,NULL);
    printf("%s",log);
    return 6;
  }
	//カーネルの作成
  printf("Creating kernel.\n");
  cl_kernel kernel=clCreateKernel(program,"place",&status);
  if(status!=CL_SUCCESS){
    return 1;
	}
	// コマンドキューの作成
	// clCreateCommandQueue()指定したデバイスのコマンドキューを作成する。
	// context    OpenCLコンテキスト。
	// device    第１引数のcontextに関連づけられたデバイス。
	// properties    コマンドキューに適用するプロパティのリスト。
	// errcode_ret    エラーコードを格納する変数。
  printf("Creating command queue.\n");
	printf("num_devices:%d\n", num_devices);
	
 	cl_command_queue cmd_queue=clCreateCommandQueue(context,devices[1],0,&status);
//	cl_command_queue cmd_queue[num_devices];
//	for(int i=0;i<(int)num_devices;i++){
// 		cmd_queue[i]=clCreateCommandQueue(context,devices[i],0,&status);
//	}
  if(status!=CL_SUCCESS){
    return 1;
	}
  // List of in-progress tasks
  struct queenState inProgress[spread]={ 0 };
  /**
   * 初期化
   */
  //for(int i=0; i < spread; i++){
  for(int i=0; i < si; i++){
    struct queenState s={ 0 };
		s.BOUND1=i;
    s.id=i;
    s.bm=(1<<si)-1;
    inProgress[i]=s;
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
    //printf("loop\n");
    cl_mem buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(inProgress),NULL,&status);
    if(status!=CL_SUCCESS){
      printf("Couldn't create buffer.\n");
      return 1;
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
      return 1;
    }
		//メモリバッファへの書き込み
    status=clEnqueueWriteBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
//		for(int i=0;i<(int)num_devices;i++){
//    	status=clEnqueueWriteBuffer(cmd_queue[i],buffer,CL_TRUE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
//		}
    if(status!=CL_SUCCESS){
      printf("Couldn't enque write buffer command.");
      return 1;
    }
		//カーネルの実行
    size_t globalSizes[]={ spread };
   	status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
//		for(int i=0;i<(int)num_devices;i++){
//   		status=clEnqueueNDRangeKernel(cmd_queue[i],kernel,1,0,globalSizes,NULL,0,NULL,NULL);
//		}

/*
    status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
    status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
    status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
    status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
    status=clEnqueueNDRangeKernel(cmd_queue,kernel,1,0,globalSizes,NULL,0,NULL,NULL);
*/
    if(status!=CL_SUCCESS){
      printf("Couldn't enque kernel execution command.");
      return 1;
    }
		//結果を読み込み
   	status=clEnqueueReadBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
//		for(int i=0;i<(int)num_devices;i++){
//    	status=clEnqueueReadBuffer(cmd_queue[i],buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
//		}
    if(status!=CL_SUCCESS){
      printf("Couldn't enque read command.");
      return 1;
    }
		//実行が終わるまで待機
    status=clFinish(cmd_queue);
//		for(int i=0;i<(int)num_devices;i++){
//    	status=clFinish(cmd_queue[i]);
//		}
    if(status!=CL_SUCCESS){
      printf("Couldn't finish command queue.");
      return 1;
    }
  }//end while

	//結果の印字
  for(int i=0; i < spread; i++){
    printf("%d: %llu\n",inProgress[i].id,inProgress[i].lTotal);
	}
  free(devices);
  return 0;
}
