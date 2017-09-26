// Without OPENCL_STYLE defined, this program will compile with gcc/clang,
// which facilitates testing and experimentation. Without it defined, it
// compiles as an OpenCL shader.

#ifndef OPENCL_STYLE
  // Declarations appropriate to this program being compiled with gcc.
  #include "stdio.h"
  #include "stdint.h"
  typedef int64_t qint;
  // A stub for OpenCL's get_global_id function.
  int get_global_id(int dimension) { return 0; }
  #define CL_KERNEL_KEYWORD
  #define CL_GLOBAL_KEYWORD
  #define CL_CONSTANT_KEYWORD
  #define CL_PACKED_KEYWORD
  #define NUM_QUEENS 14
#else
  // Declarations appropriate to this program being compiled as an OpenCL
  // kernel. OpenCL has a 64 bit long and requires special keywords to designate
  // where and how different objects are stored in memory.
  typedef long qint;
  typedef long int64_t;
  typedef ulong uint64_t;
  typedef ushort uint16_t;
  #define CL_KERNEL_KEYWORD __kernel
  #define CL_GLOBAL_KEYWORD __global
  #define CL_CONSTANT_KEYWORD __constant
  #define CL_PACKED_KEYWORD  __attribute__ ((packed))
#endif

enum { PLACE, REMOVE, DONE };

// State of individual computation
struct CL_PACKED_KEYWORD queenState
{
  int id;
  qint masks[NUM_QUEENS];
  uint64_t solutions; // Number of solutinos found so far.
  char step;
  char col;
  char startCol; // First column this individual computation was tasked with filling.
  qint mask;
  qint rook;
  qint add;
  qint sub;
};

CL_CONSTANT_KEYWORD const qint dodge = (1 << NUM_QUEENS) - 1;

CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState * state)
{
  //inProgress配列から1個取り出す
  //CPU->GPUに渡す値は配列になる
  //ワークアイテムを識別する数値を取得する
  //get_global_idはグローバルなワークアイテムID
  //取得した数値に対応するデータに対して処理を行う
  int index = get_global_id(0);
  int si=NUM_QUEENS;
  printf("si:%d",si);
  int fA[2*NUM_QUEENS-1]; //fA:flagA 縦 配置フラグ　
  int fB[2*NUM_QUEENS-1];  //fB:flagB 斜め配置フラグ　
  int fC[2*NUM_QUEENS-1];  //fC:flagC 斜め配置フラグ　
  int aB[NUM_QUEENS];      //aB:aBoard[] チェス盤の横一列
  int r=0;
  for(int j=0;j<si;j++){ aB[j]=j; } //aBを初期化
  uint64_t Total=0;
  if(r==si){
    Total++; //解を発見
  }else{
    for(int i=0;i<si;i++){
      aB[r]=i ;
      //バックトラック 制約を満たしているときだけ進む
      if(fA[i]==0&&fB[r-i+(si-1)]==0&&fC[r+i]==0){
        fA[i]=fB[r-aB[r]+si-1]=fC[r+aB[r]]=1; 
        //NQueen(r+1,si);//再帰
        fA[i]=fB[r-aB[r]+si-1]=fC[r+aB[r]]=0; 
      }
    }  
  }
  state[index].solutions = Total;
  state[index].step = DONE;
}

