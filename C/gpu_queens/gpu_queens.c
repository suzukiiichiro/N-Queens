
#include "stdio.h"
#include "string.h"
#ifdef __APPLE__ // MacOSであるかを確認
#include "OpenCL/cl.h" //MacOSの場合はインクルード
#else
#include <CL/cl.h> //Windows/Unix/Linuxの場合はインクルード
#endif

#define PROGRAM_FILE "./queen_kernel.c" //カーネルソースコード
#define FUNC "place" //カーネル関数の名称を設定
#include "time.h"
#include "sys/time.h"
#define BUFFER_SIZE 4096

const int32_t si=13;
long lGTotal;
long lGUnique;
typedef int64_t qint;
enum { Place,Remove,Done };
struct queenState {
	int BOUND1;
  int id;
  qint aB[si];
  uint64_t lTotal;
  char step;
  int y;
  int startCol;
  qint bm;
  qint down;
  qint right;
  qint left;
} __attribute__((packed));
void get_queens_code(char ** buffer);
int all_tasks_done(struct queenState * tasks,size_t num_tasks);
/**
main()OpenCL 主な流れ 
	clGetPlatformIDs();         // プラットフォーム一覧を取得
	clGetDeviceIDs();           // デバイス一覧を取得
	clCreateContext();          // コンテキストの作成
	clCreateCommandQueue();     // コマンドキュー作成
	clCreateProgramWithSource();// ソースコードからカーネルプログラム作成
	clBuildProgram();						// カーネルプログラムのビルド
	clGetProgramBuildInfo();		// プログラムのビルド情報を取得
	clCreateKernel();						// カーネルの作成
	clCreateBuffer();           // 書き込み・読み込みメモリバッファの作成
	clSetKernelArg();           // カーネル引数の設定
	clEnqueueWriteBuffer();     // メモリバッファへの書き込み
	clEnqueueNDRangeKernel();   // カーネル実行
	clEnqueueReadBuffer();      // メモリバッファから結果読み出し
	clFinish();                 // 実行が終わるまで待機
	clReleaseMemObject();
	clReleaseKernel();
	clReleaseProgram();
	clReleaseCommandQueue();
	clReleaseContext();
*/
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
/**
 * タスクの終了を待機する
 */
int all_tasks_done(struct queenState * tasks,size_t num_tasks){
  for(int i=0; i <(int)num_tasks; i++)
    if(tasks[i].step==Done)
      return 1;
  return 0;
}
int main(){
  struct timeval t0;
  struct timeval t1;
  gettimeofday(&t0, NULL);
	char value[BUFFER_SIZE];
	size_t size;
  cl_platform_id platform; 	//プラットフォーム
 	cl_device_id devices;			//デバイス一覧
  cl_uint num_devices;			//デバイス数
	cl_context context;				//コンテキスト
  cl_program program;				//プログラム
	cl_command_queue cmd_queue;//コマンドキュー
	cl_kernel kernel;					//カーネル
	/**
	プラットフォーム一覧を取得
	現在利用可能なOpenCLのプラットフォームの情報を取得
	clGetPlatformIDs()使用できるプラットフォームの数とID一覧を取得する関数
	numEntries:追加可能なエントリーの数
	platforms : 見つかったプラットフォームの一覧が代入されるポインタ
	numPlatforms : 使用できるプラットフォームの数が代入されるポインタ  
	戻り値　CL_SUCCESS 成功 CL_INVALID_VALUE 失敗
  */
  cl_int status=clGetPlatformIDs(1,&platform,NULL); //プラットフォームを取得
	status=clGetPlatformInfo(platform,CL_PLATFORM_NAME,BUFFER_SIZE,value,&size);
 	printf("CL_PLATFORM_NAME:%s\n",value);
	status=clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,value,&size);	
  printf("CL_PLATFORM_VERSION:%s\n",value);
  if(status!=CL_SUCCESS){ printf("Couldn't get platform info."); return 1; }
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
 	devices=malloc(num_devices * sizeof(cl_device_id));
 	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,num_devices,&devices,&num_devices);
	if(status==CL_DEVICE_NOT_FOUND){
 		status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,num_devices,&devices,&num_devices);
		printf("CL_DEVICE_TYPE_ALL\n");
	}else{ printf("CL_DEVICE_TYPE_GPU\n"); }
  if(status!=CL_SUCCESS){ printf("Couldn't get device count."); return 2; 
  }else{ printf("CL_DEVICE COUNT:%d\n",num_devices); }
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
  context=clCreateContext(NULL,num_devices,&devices,NULL,NULL,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating context.\n"); return 4; 
  }else{ printf("Creating context.\n"); }
	/**
	コマンドキューの生成
	clCreateCommandQueue()指定したデバイスのコマンドキューを作成する。
	context   コンテキスト。
	device    第１引数のcontextに関連づけられたデバイス。
	properties    コマンドキューに適用するプロパティのリスト。
	errcode_ret    エラーコードを格納する変数。
	*/
 	cmd_queue=clCreateCommandQueue(context,devices,0,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating command queue."); return 5; 
  }else{ printf("Creating command queue.\n"); }
	/** ソースコードからカーネルコードプログラムを作成 */
  char *code;
  get_queens_code(&code);
  if(code==NULL){ printf("Couldn't load the code."); return 5;
  }else{ printf("Loading kernel code.\n"); }
  program=clCreateProgramWithSource(context,1,(const char **)&code,NULL,&status);
  free(code);
  if(status!=CL_SUCCESS){ printf("Couldn't creating program."); return 6; 
  }else{ printf("Creating program.\n"); }
	/**
	プログラムのビルド
	clBuildProgram()カーネルオブジェクトを作成する。
	program    実行ファイルを作成するもとになるプログラム
	kernel_name    __kernelで事前に指定する関数名。
	errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
	*/
  status=clBuildProgram(program,num_devices,&devices,NULL,NULL,NULL);
  if(status!=CL_SUCCESS){
    char log[2048];
   	status=clGetProgramBuildInfo(program,devices,CL_PROGRAM_BUILD_LOG,2048,log,NULL);
    printf("%s",log); printf("Couldn't building program."); return 7;
  }else{ printf("Building program.\n"); }
	/** カーネルオブジェクトの宣言 */
  kernel=clCreateKernel(program,FUNC,&status);
  if(status!=CL_SUCCESS){ printf("Couldn't creating kernel."); return 8; 
  }else{ printf("Creating kernel.\n"); }
  /**
   * 初期化 カーネルオブジェクトに渡す構造体
   */
  struct queenState inProgress[si]={0};
  for(int i=0; i <si; i++){
    struct queenState s={0};
		s.BOUND1=i; //BOUND1の初期化
    s.id=i;
    s.bm=(1<<si)-1;
		for (int i=0; i < si; i++){ s.aB[i]=i; }
		s.lTotal=0;
		s.step=0;
		s.y=0;
		s.startCol =1;
		s.down=0;
		s.right=0;
		s.left=0;
    inProgress[i]=s;
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
  printf("Starting computation of Q(%d)\n",si);
  while(!all_tasks_done(inProgress,si)){
    cl_mem buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(inProgress),NULL,&status);
    if(status!=CL_SUCCESS){ printf("Couldn't create buffer.\n"); return 9; }
		/** メモリバッファへの書き込み */
    status=clEnqueueWriteBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),&inProgress,0,NULL,NULL);
    if(status!=CL_SUCCESS){ printf("Couldn't enque write buffer command."); return 10; }
		/**
    カーネルの引数をセット
    clSetKernelArg()カーネルの特定の引数に値をセットする。
    kernel    値をセットするカーネル。
    arg_index    引数のインデックス。
    arg_size    引数として渡すのデータのサイズ。
    arg_value    第２引数arg_indexで指定した引数にわたすデータへのポインタ。
		*/
    status=clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
    if(status!=CL_SUCCESS){ printf("Couldn't set kernel arg."); return 11; }
		//カーネルの実行 カーネルを実行するコマンドをキューに入れて、カーネル関数をデバイスで実行
		/** 
     * 設定項目
		 */
		size_t dim=1;
		size_t global_offset[]={0};
    size_t global_work_size[]={si};
		size_t local_work_size[]={si};
   	status=clEnqueueNDRangeKernel(cmd_queue,kernel,dim,global_offset,global_work_size,local_work_size,0,NULL,NULL);
    if(status!=CL_SUCCESS){ printf("Couldn't enque kernel execution command."); return 12; }
		//結果を読み込み
   	status=clEnqueueReadBuffer(cmd_queue,buffer,CL_TRUE,0,sizeof(inProgress),inProgress,0,NULL,NULL);
    if(status!=CL_SUCCESS){ printf("Couldn't enque read command."); return 13; }
		//実行が終わるまで待機
    status=clFinish(cmd_queue);
    if(status!=CL_SUCCESS){ printf("Couldn't finish command queue."); return 14; }
		//解放
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(cmd_queue);
		clReleaseContext(context);
  }//end while
	//結果の印字
  long lGTotal=0;
	long lGUnique=0;
  for(int i=0; i <si; i++){
    printf("%d: %llu\n",inProgress[i].id,inProgress[i].lTotal);
    lGTotal+=inProgress[i].lTotal;
	}
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
  printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", si,lGTotal,lGUnique,dd,hh,mm,ss,ms); 
  return 0;
}
