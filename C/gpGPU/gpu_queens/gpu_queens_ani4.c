#include "stdio.h"
#include "string.h"
#include "OpenCL/cl.h"

const int32_t numQueens = 15;
const int32_t spread = 15;
typedef int64_t qint;
uint64_t lGTotal;
uint64_t lGUnique;
enum {
	Place, Remove, Done
};
struct queenState {
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
  qint BOUND1;
}__attribute__((packed));
/**
 * カーネルコードの読み込み
 */
void get_queens_code(char ** buffer) {
	char prefix[256];
	int prefixLength = snprintf(prefix, 256, "#define OPENCL_STYLE\n#define NUM_QUEENS %d\n", numQueens);
	FILE * f = fopen("./queen_kernel.c", "rb");
	if (!f) {
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

int all_tasks_done(struct queenState * tasks, size_t num_tasks) {
	for (int i = 0; i < num_tasks; i++)
		if (tasks[i].step == Done)
			return 1;
	return 0;
}
/**
main() OpenCL 主な流れ 

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
int main() {
	//
	cl_int status;

	// プラットフォーム一覧を取得
	// clGetPlatformIDs()使用できるプラットフォームの数とID一覧を取得する関数
	// 戻り値　CL_SUCCESS 成功 CL_INVALID_VALUE 失敗
	// PLATFORM : 見つかったプラットフォームの最大取得数
	// platforms : 見つかったプラットフォームの一覧が代入されるポインタ
	// &platformCount : 使用できるプラットフォームの数が代入されるポインタ  
	//status = clGetPlatformIDs(1, &platform, NULL);
	printf("Looking up first platform.\n");
	cl_platform_id platform;
	cl_uint platformCount;
	status = clGetPlatformIDs(1, &platform, &platformCount);
	if(status != CL_SUCCESS) { return 1; }
	if(platformCount==0){ printf("No platform.\n"); return EXIT_FAILURE; }
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
	printf("Determining number of devices on platform.\n");
	cl_uint deviceCount;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	if(status != CL_SUCCESS){ return 2;}
	if(deviceCount==0){
		printf("No device.\n");
		return EXIT_FAILURE;
	}
	printf("Getting device IDs.\n");
	cl_device_id * devices = malloc(deviceCount * sizeof(cl_device_id));
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
	printf("%d devices detected.\n", deviceCount);
	if (status != CL_SUCCESS){ return 3;}
	// コンテキストの作成
	// clCreateContext() ひとつ以上のデバイスで使用するためのコンテキストを作成する。
	// nullptr コンテキストプロパティを指定する。
	// 各プロパティ名にはそれぞれに対応した要求される値が続く。この一覧の終端には0がつけ
	// られる。引数porpertiesには、処理依存のプラットフォームの場合に限りNULLを指定する
	// ことができる。
	// 1 : 第３引数devicesで指定されたデバイスの数
	// devices : 一意に定まる、clGetDeviceIDs関数で取得されたデバイス、また
	// はclCreateSubDevicesで作成されたサブデバイス。
	// nullptr : アプリケーションによって登録することが可能なコールバック関数。
	// nullptr : 引数pfn_notifyで設定したコールバック関数が呼び出されたとき、データが
	// 渡されるポインタ。この引数はNULLにした場合、無視される
	// &err エラーが発生した場合、そのエラーに合わせたエラーコードが返される。
	printf("Creating context.\n");
	cl_context context = clCreateContext(NULL, deviceCount, devices, NULL, NULL, &status);
	if (status != CL_SUCCESS){ return 4;}
	printf("Loading kernel code.\n");
	char * code;
	get_queens_code(&code);
	if (code == NULL) { printf("Couldn't load the code."); }
	//ソースコードからカーネルプログラム作成
	printf("Creating program.\n");
	cl_program program = clCreateProgramWithSource(context, 1,
			(const char **) &code, NULL, &status);
	free(code);
	if (status != CL_SUCCESS)
		return 5;
	// プログラムのビルド
	// clBuildProgram() カーネルオブジェクトを作成する。
	// program    実行ファイルを作成するもとになるプログラム
	// kernel_name    __kernelで事前に指定する関数名。
	// errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
	printf("Building program.\n");
	status = clBuildProgram(program, deviceCount, devices, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		char log[2048];
		status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 2048, log, NULL);
		printf("%s", log);
		return 6;
	}
	//カーネルの作成
	printf("Creating kernel.\n");
	cl_kernel kernel = clCreateKernel(program, "place", &status);
	if (status != CL_SUCCESS){ return 1;}
	// コマンドキューの作成
	// clCreateCommandQueue() 指定したデバイスのコマンドキューを作成する。
	// context    OpenCLコンテキスト。
	// device    第１引数のcontextに関連づけられたデバイス。
	// properties    コマンドキューに適用するプロパティのリスト。
	// errcode_ret    エラーコードを格納する変数。
	printf("Creating command queue.\n");
	cl_command_queue cmd_queue = clCreateCommandQueue(context, devices[1], 0, &status);
	if (status != CL_SUCCESS){ return 1;}
	// List of in-progress tasks
	struct queenState inProgress[spread] = { 0 };
	for (int i = 0; i < spread; i++) {
		struct queenState s = { 0 };
		s.id = i;
    s.startCol=1;
		s.mask = (1 << numQueens) - 1;
    s.BOUND1=i;
		inProgress[i] = s;
	}
	// デバイスメモリを確保しつつデータをコピー
	// clCreateBuffer() バッファオブジェクトを作成する。
	// context バッファオブジェクトを作成するために必要なOpenCLコンテキスト。
	// flags    「バッファオブジェクトをどのようなメモリ領域に割り当てるか」「メモリ領域
	// をどのように使用するか」のような割り当てやusageに関する情報を指定するビットフィールド。
	// CL_MEM_READ_WRITE カーネルにメモリ領域へのRead/Writeを許可する設定。
	// CL_MEM_USE_HOST_PTR デバイスメモリ内でhost_ptrを指定することにより、OpsnCL処理に
	// バッファをキャッシュすることを許可する。
	// size    割り当てられたバッファメモリオブジェクトのバイトサイズ
	// host_ptr    アプリケーションにより既に割り当てられているバッファデータへのポインタ。
	// errcode_ret    実行結果に関連づけられたエラーコードを格納するポインタ。
	printf("Starting computation of Q(%d)\n", numQueens);
	while (!all_tasks_done(inProgress, spread)) {
		printf("loop\n");
		cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(inProgress), NULL, &status);
		if (status != CL_SUCCESS) {
			printf("Couldn't create buffer.\n");
			return 1;
		}
    // カーネルの引数をセット
    // clSetKernelArg() カーネルの特定の引数に値をセットする。
    // kernel    値をセットするカーネル。
    // arg_index    引数のインデックス。
    // arg_size    引数として渡すのデータのサイズ。
    // arg_value    第２引数arg_indexで指定した引数にわたすデータへのポインタ。
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
		if (status != CL_SUCCESS) {
			printf("Couldn't set kernel arg.");
			return 1;
		}
		//メモリバッファへの書き込み
		status = clEnqueueWriteBuffer(cmd_queue, buffer, CL_TRUE, 0, sizeof(inProgress), &inProgress, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("Couldn't enque write buffer command.");
			return 1;
		}
		//カーネルの実行
		size_t globalSizes[] = { spread };
		status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
//		status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
//		status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
//		status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
//		status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, 0, globalSizes, NULL, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("Couldn't enque kernel execution command.");
			return 1;
		}
		//結果を読み込み
		status = clEnqueueReadBuffer(cmd_queue, buffer, CL_TRUE, 0, sizeof(inProgress), inProgress, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("Couldn't enque read command.");
			return 1;
		}
		//実行が終わるまで待機
		status = clFinish(cmd_queue);
		if (status != CL_SUCCESS) {
			printf("Couldn't finish command queue.");
			return 1;
		}
	}
	//結果の印字
  	lGTotal=0;
  	lGUnique=0;
	for (int i = 0; i < spread; i++){
		printf("%d: %llu\n", inProgress[i].id, inProgress[i].solutions);
    		lGTotal+=inProgress[i].solutions;
	}
	printf("lGTotal:%llu\n",lGTotal);
	free(devices);
	return 0;
}
