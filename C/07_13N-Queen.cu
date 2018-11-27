/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 必要なこと
　１．$ lspci | grep -i  nvidia 
      で、nVidiaが存在しなければ絶対動かない。
　２．$ nvidia-smi
　　　で、以下の通りに出力され、ＧＰＵの存在が確認できなければ絶対に動かない。

bash-4.2$ nvidia-smi
Wed Jun 27 02:36:34 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.111                Driver Version: 384.111                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   38C    P8    30W / 149W |      1MiB / 11439MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+


　３．nVidia CUDAのインストールは思った以上に難しい。
　　　特にMacはOSのバージョン、xcode、command line toolsとの相性もある。
     
  手軽なのは、有料だが Amazon AWSのEC2でCUDA nVidia GPU 対応のサーバーを使うこと。
　（こちらが絶対的におすすめ）


  ４．ここまでが了解であれば、以下のコマンドで実行可能だ。


 コンパイルと実行

 # CPUだけの実行
 $ nvcc gpuNQueen.cu -o gpuNQueen && ./gpuNQueen -cpu 

 # GPUだけの実行
 $ nvcc gpuNQueen.cu -o gpuNQueen && ./gpuNQueen -gpu

 # CPUとGPUの実行
 $ nvcc gpuNQueen.cu -o gpuNQueen && ./gpuNQueen



 １３．ＧＰＵ nVidia-CUDA               N17=    1.67
 *
 *  実行結果
CPU
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.00
12:             14200                 0          00:00:00:00.00
13:             73712                 0          00:00:00:00.05
14:            365596                 0          00:00:00:00.29
15:           2279184                 0          00:00:00:01.94

GPU
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.06
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.00
12:             14200                 0          00:00:00:00.01
13:             73712                 0          00:00:00:00.02
14:            365596                 0          00:00:00:00.01
15:           2279184                 0          00:00:00:00.05
16:          14772512                 0          00:00:00:00.27
17:          95815104                 0          00:00:00:01.65

*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define THREAD_NUM		96
#define MAX 15

long Total=0 ;        //合計解
long Unique=0;
int down[2*MAX-1]; //down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //left:flagB 斜め配置フラグ　
int right[2*MAX-1];  //right:flagC 斜め配置フラグ　
int aBoard[2*MAX-1];      //aBoard[] チェス盤の横一列

/**
  case 1 : 再帰　非CUDA 07_04相当
 07_04
 15:      2279184               0           13.50
  1. バックトラック

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.01
12:             14200                 0          00:00:00:00.08
13:             73712                 0          00:00:00:00.45
14:            365596                 0          00:00:00:02.72
15:           2279184                 0          00:00:00:17.61
**/
void solve_nqueen_Recursive_BT(int row,int size){
  if(row==size){
    Total++;
  }else{
    for(int i=0;i<size;i++){
      aBoard[row]=i ;
      if(down[i]==0&&left[row-i+(size-1)]==0&&right[row+i]==0){
        down[i]=left[row-aBoard[row]+size-1]=right[row+aBoard[row]]=1; 
        solve_nqueen_Recursive_BT(row+1,size);
        down[i]=left[row-aBoard[row]+size-1]=right[row+aBoard[row]]=0; 
      }
    }  
  }
}

/**
  case 2 : 非再帰　非CUDA
  1. バックトラック

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.01
12:             14200                 0          00:00:00:00.07
13:             73712                 0          00:00:00:00.44
14:            365596                 0          00:00:00:02.72
15:           2279184                 0          00:00:00:17.41
**/
void solve_nqueen_nonRecursive_BT(int r,int n){
  bool matched;
  while(r>=0) {
    matched=false;
    for(int i=aBoard[r]+1;i<n;i++) {
      if(0==down[i] && 0==left[r+(n-1)-i] && 0==right[r+i]) {
        if(aBoard[r] >= 0) {
          down[aBoard[r]]=left[r+(n-1)-aBoard[r]]=right[r+aBoard[r]]=0;
        }
        aBoard[r]=i;
        down[i]=left[r+(n-1)-i]=right[r+i]=1;
        matched=true;
        break;
      }
    }
    if(matched){
      r++;
      if(r==n){
        Total++;
        r--;
      }
    }else{
      if(aBoard[r]>=0){
        int tmp=aBoard[r];
        aBoard[r]=-1;
        down[tmp]=left[r+(n-1)-tmp]=right[r+tmp]=0;
      }
      r--;
    }
  }
}
/** 
  case 3 : 再帰 非CUDA
  1. バックトラック
  2. ビットマップ

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
13:             73712                 0          00:00:00:00.07
14:            365596                 0          00:00:00:00.45
15:           2279184                 0          00:00:00:02.81
*/
long long solve_nqueen_Recursive_BT_BM(int size,unsigned int left,unsigned int down,unsigned int right) {
  unsigned int mask=(1<<size)-1;
  if(down==mask){
    return 1;
  }
	unsigned int bitmap=(left|down|right);
	if((bitmap&mask)==mask){
    return 0;
  }
	long long total=0;
	unsigned int bit=(bitmap+1)&~bitmap;
	while(bit&mask){
		total+=solve_nqueen_Recursive_BT_BM(size,(left|bit)<<1,down|bit,(right|bit)>>1);
		bitmap|=bit;
		bit=(bitmap+1)&~bitmap;
	}
	return total;
}

/** 
  case 4 : 非再帰 非CUDA
  1. バックトラック
  2. ビットマップ

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 0          00:00:00:00.00
 5:                10                 0          00:00:00:00.00
 6:                 4                 0          00:00:00:00.00
 7:                40                 0          00:00:00:00.00
 8:                92                 0          00:00:00:00.00
 9:               352                 0          00:00:00:00.00
10:               724                 0          00:00:00:00.00
11:              2680                 0          00:00:00:00.00
12:             14200                 0          00:00:00:00.00
13:             73712                 0          00:00:00:00.04
14:            365596                 0          00:00:00:00.22
15:           2279184                 0          00:00:00:01.49
*/
long long solve_nqueen_nonRecursive_BT_BM(int n){
  unsigned int down[32];unsigned int left[32];unsigned int right[32];unsigned int bm[32];
  if(n<=0||n>32){return 0;}
  const unsigned int msk=(1<<n)-1;long long total=0;long long uTotal=0;
  int i=0;int j=0;unsigned int bit;
  down[0]=0;left[0]=0;right[0]=0;bm[0]=0;
  for(j=0;j<(n+1)/2;j++){
    bit=(1<<j);
    bm[0]|=bit;down[1]=bit;left[1]=bit<<1;right[1]=bit>>1;
    bm[1]=(down[1]|left[1]|right[1]);
    i=1;
    if(n%2==1&&j==(n+1)/2-1){uTotal=total;total=0;}
    while(i>0){
      if((bm[i]&msk)==msk){i--;}
      else{
        bit=((bm[i]+1)^bm[i])&~bm[i];
        bm[i]|=bit;
        if((bit&msk)!=0){
          if(i+1==n){total++;i--;}
          else{
            down[i+1]=down[i]|bit;left[i+1]=(left[i]|bit)<<1;right[i+1]=(right[i]|bit)>>1;
            bm[i+1]=(down[i+1]|left[i+1]|right[i+1]);
            i++;
          }
        }else{i--;}
      }
    }
  }
  if(n%2==0){return total*2;}
  else{return uTotal*2+total;}
}

/**
  case 5 : 再帰 非CUDA 07_08相当
  07_08
  15:      2279184          285053            5.88

  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO

5. 再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.02
13:             73712              9233          00:00:00:00.14
14:            365596             45752          00:00:00:00.83
15:           2279184            285053          00:00:00:05.61
*/
int aT[MAX];
int aS[MAX];
int bit;
int C2=0;
int C4=0;
int C8=0;

long getTotal();
long getUnique();
void rotate_bitmap(int bf[],int af[],int si);
void vMirror_bitmap(int bf[],int af[],int si);
int rh(int a,int sz);
int intncmp(int lt[],int rt[],int si);
void symmetryOps_bm(int si);

void solve_nqueen_Recursive_BT_BM_SO(int size,int mask,int row,int left,int down,int right){
  int bitmap=mask&~(left|down|right); //配置可能フィールド
  if(row==size){
    if(!bitmap){
      aBoard[row]=bitmap;
      symmetryOps_bm(size);
    }
  }else{
    while(bitmap) {
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      solve_nqueen_Recursive_BT_BM_SO(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  } 
}

long getUnique(){ 
  return C2+C4+C8;
}
long getTotal(){ 
  return C2*2+C4*4+C8*8;
}
void rotate_bitmap(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
  }
}
void vMirror_bitmap(int bf[],int af[],int si){
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
}
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
int intncmp(int lt[],int rt[],int si){
  int rtn=0;
  for(int k=0;k<si;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
void symmetryOps_bm(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aBoard[i];}
  rotate_bitmap(aT,aS,si);    //時計回りに90度回転
  int k=intncmp(aBoard,aS,si);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aS,aT,si);  //時計回りに180度回転
    k=intncmp(aBoard,aT,si);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aT,aS,si);//時計回りに270度回転
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aS[i]=aBoard[i];}
  vMirror_bitmap(aS,aT,si);   //垂直反転
  k=intncmp(aBoard,aT,si);
  if(k>0){ return; }
  if(nEquiv>2){               //-90度回転 対角鏡と同等       
    rotate_bitmap(aT,aS,si);
    k=intncmp(aBoard,aS,si);
    if(k>0){return;}
    if(nEquiv>4){             //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aBoard,aT,si);
      if(k>0){ return;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){ C2++; }
  if(nEquiv==4){ C4++; }
  if(nEquiv==8){ C8++; }
}


/**
  case 6 : 非再帰 非CUDA
  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO

6. 非再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.03
13:             73712              9233          00:00:00:00.19
14:            365596             45752          00:00:00:01.14
15:           2279184            285053          00:00:00:07.70
*/
struct HIKISU{
  int Y;
  int I;
  int M;
  int L;
  int D;
  int R;
  int B;
};
struct STACK {
  struct HIKISU param[MAX];
  int current;
};
void solve_nqueen_nonRecursive_BT_BM_SO(int n,int msk,int y,int l,int d,int r){
  struct STACK stParam;
  for (int m=0;m<n;m++){ 
    stParam.param[m].Y=0;
    stParam.param[m].I=n;
    stParam.param[m].M=0;
    stParam.param[m].L=0;
    stParam.param[m].D=0;
    stParam.param[m].R=0;
    stParam.param[m].B=0;
  }
  stParam.current=0;
  int bend=0;
  int rflg=0;
  int bm;
  while(1){
  if(rflg==0){
   bm=msk&~(l|d|r); //配置可能フィールド
  }
  if(y==n&&!bm&&rflg==0){
    aBoard[y]=bm;
    symmetryOps_bm(n);
  }else{
    while(bm|| rflg==1) {
        if(rflg==0){
      bm^=aBoard[y]=bit=(-bm&bm); //最も下位の１ビットを抽出
          if(stParam.current<MAX){
            stParam.param[stParam.current].Y=y;
            stParam.param[stParam.current].I=n;
            stParam.param[stParam.current].M=msk;
            stParam.param[stParam.current].L=l;
            stParam.param[stParam.current].D=d;
            stParam.param[stParam.current].R=r;
            stParam.param[stParam.current].B=bm;
            (stParam.current)++;
          }
          y=y+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend=1;
          break;
        }
        if(rflg==1){ 
          if(stParam.current>0){
            stParam.current--;
          }
          n=stParam.param[stParam.current].I;
          y=stParam.param[stParam.current].Y;
          msk=stParam.param[stParam.current].M;
          l=stParam.param[stParam.current].L;
          d=stParam.param[stParam.current].D;
          r=stParam.param[stParam.current].R;
          bm=stParam.param[stParam.current].B;
          rflg=0;
        }
    }
      if(bend==1 && rflg==0){
        bend=0;
        continue;
      }
  } 
    if(y==0){
      break;
    }else{
      //goto ret;
      rflg=1;
    }
  }
}

/**
  case 7 : 再帰 非CUDA 07_09に相当
  07_09
  15:      2279184          285053            3.33

  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
  4. 最上段のクイーンの位置による枝刈り BOUND 

7. 再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)＋枝刈り(BOUND)
 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.01
13:             73712              9233          00:00:00:00.08
14:            365596             45752          00:00:00:00.52
15:           2279184            285053          00:00:00:03.42

*/
int BOUND1;
int BOUND2;
int TOPBIT;
int ENDBIT;
int SIDEMASK;
int LASTMASK;

void backTrack2_Recursive_BT_BM_SO_BOUND(int size,int mask,int row,int l,int d,int r){
  int bit;
  int bitmap=mask&~(l|d|r); /* 配置可能フィールド */
  if (row==size) {
    if(!bitmap){
      aBoard[row]=bitmap;
      symmetryOps_bm(size);
    }
  }else{
    while(bitmap) {
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      backTrack2_Recursive_BT_BM_SO_BOUND(size,mask,row+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
void backTrack1_Recursive_BT_BM_SO_BOUND(int size,int mask,int row,int l,int d,int r){
  int bit;
  int bitmap=mask&~(l|d|r); /* 配置可能フィールド */
  if (row==size) {
    if(!bitmap){
      aBoard[row]=bitmap;
      symmetryOps_bm(size);
    }
  }else{
    while(bitmap) {
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //SO最も下位の１ビットを抽出
      backTrack1_Recursive_BT_BM_SO_BOUND(size,mask,row+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
void solve_nqueen_Recursive_BT_BM_SO_BOUND(int size,int mask) {
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    backTrack1_Recursive_BT_BM_SO_BOUND(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2_Recursive_BT_BM_SO_BOUND(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}

/**
  case 8 : 非再帰 非CUDA
  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
  4. 最上段のクイーンの位置による枝刈り BOUND

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.02
13:             73712              9233          00:00:00:00.10
14:            365596             45752          00:00:00:00.67
15:           2279184            285053          00:00:00:04.22
*/
//long long solve_nqueen_nonRecursive_BT_BM_SO_BOUND(int n){
//  return true;
//}
void backTrack2_nonRecursive_BT_BM_SO_BOUND(int is,int msk,int y, int l, int d, int r);
void backTrack1_nonRecursive_BT_BM_SO_BOUND(int si,int msk,int y, int l, int d, int r);
void solve_nqueen_nonRecursive_BT_BM_SO_BOUND(int n,int B1,int B2,int msk){
  int bit;
  if(B1==0){
    aBoard[0]=1;
    for(BOUND1=2;BOUND1<n-1;BOUND1++){
      aBoard[1]=bit=(1<<BOUND1);
      backTrack1_nonRecursive_BT_BM_SO_BOUND(n,msk,2,(2|bit)<<1,(1|bit),(bit>>1));
    }
  } else{
    BOUND1=B1;
    BOUND2=B2;
    if(BOUND1<BOUND2){
      aBoard[0]=bit=(1<<BOUND1);
      backTrack2_nonRecursive_BT_BM_SO_BOUND(n,msk,1,bit<<1,bit,bit>>1);
    }
  }
}
void backTrack2_nonRecursive_BT_BM_SO_BOUND(int si,int msk,int y,int l,int d,int r){
  struct STACK stParam_2;
  for (int m=0;m<si;m++){ 
    stParam_2.param[m].Y=0;
    stParam_2.param[m].I=si;
    stParam_2.param[m].M=0;
    stParam_2.param[m].L=0;
    stParam_2.param[m].D=0;
    stParam_2.param[m].R=0;
    stParam_2.param[m].B=0;
  }
  stParam_2.current=0;
  int bend_2=0;
  int rflg_2=0;
  int bit;
  int bm;
  while(1){
//start:
    if(rflg_2==0){
      bm=msk&~(l|d|r); /* 配置可能フィールド */
    }
    if (y==si&&rflg_2==0) {
      if(!bm){
        aBoard[y]=bm;
        symmetryOps_bm(si);
      }
    }else{
      while(bm|| rflg_2==1) {
        if(rflg_2==0){
          bm^=aBoard[y]=bit=(-bm&bm); //最も下位の１ビットを抽出
          if(stParam_2.current<MAX){
            stParam_2.param[stParam_2.current].Y=y;
            stParam_2.param[stParam_2.current].I=si;
            stParam_2.param[stParam_2.current].M=msk;
            stParam_2.param[stParam_2.current].L=l;
            stParam_2.param[stParam_2.current].D=d;
            stParam_2.param[stParam_2.current].R=r;
            stParam_2.param[stParam_2.current].B=bm;
            (stParam_2.current)++;
          }
          y=y+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_2=1;
          break;
        }
        if(rflg_2==1){ 
          if(stParam_2.current>0){
            stParam_2.current--;
          }
          si=stParam_2.param[stParam_2.current].I;
          y=stParam_2.param[stParam_2.current].Y;
          msk=stParam_2.param[stParam_2.current].M;
          l=stParam_2.param[stParam_2.current].L;
          d=stParam_2.param[stParam_2.current].D;
          r=stParam_2.param[stParam_2.current].R;
          bm=stParam_2.param[stParam_2.current].B;
          rflg_2=0;
        }
      }
      if(bend_2==1 && rflg_2==0){
        bend_2=0;
        continue;
      }
    } 
    if(y==1){
      break;
    }else{
      rflg_2=1;
    }
  }
}
void backTrack1_nonRecursive_BT_BM_SO_BOUND(int si,int msk,int y,int l,int d,int r){
  struct STACK stParam_1;
  for (int m=0;m<si;m++){ 
    stParam_1.param[m].Y=0;
    stParam_1.param[m].I=si;
    stParam_1.param[m].M=0;
    stParam_1.param[m].L=0;
    stParam_1.param[m].D=0;
    stParam_1.param[m].R=0;
    stParam_1.param[m].B=0;
  }
  stParam_1.current=0;
  int bend_1=0;
  int rflg_1=0;
  int bit;
  int bm;
  while(1){
    if(rflg_1==0){
      bm=msk&~(l|d|r); /* 配置可能フィールド */
    }
    if (y==si&&rflg_1==0) {
      if(!bm){
        aBoard[y]=bm;
        symmetryOps_bm(si);
      }
    }else{
      while(bm|| rflg_1==1) {
        if(rflg_1==0){
          bm^=aBoard[y]=bit=(-bm&bm); //最も下位の１ビットを抽出
          if(stParam_1.current<MAX){
            stParam_1.param[stParam_1.current].Y=y;
            stParam_1.param[stParam_1.current].I=si;
            stParam_1.param[stParam_1.current].M=msk;
            stParam_1.param[stParam_1.current].L=l;
            stParam_1.param[stParam_1.current].D=d;
            stParam_1.param[stParam_1.current].R=r;
          stParam_1.param[stParam_1.current].B=bm;
            (stParam_1.current)++;
          }
          y=y+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_1=1;
          break;
        }
//ret:
        if(rflg_1==1){ 
        if(stParam_1.current>0){
          stParam_1.current--;
        }
        si=stParam_1.param[stParam_1.current].I;
        y=stParam_1.param[stParam_1.current].Y;
        msk=stParam_1.param[stParam_1.current].M;
        l=stParam_1.param[stParam_1.current].L;
        d=stParam_1.param[stParam_1.current].D;
        r=stParam_1.param[stParam_1.current].R;
        bm=stParam_1.param[stParam_1.current].B;
          rflg_1=0;
        }
      }
      if(bend_1==1 && rflg_1==0){
        bend_1=0;
        continue;
      }
    } 
    if(y==2){
      break;
    }else{
      rflg_1=1;
    }
  }
}
/**
  case 9 : 再帰 非CUDA 07_10相当
  07_10
  15:      2279184          285053            1.48

  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
  4. 最上段のクイーンの位置による枝刈り BOUND 
  5. BOUNDの枝刈り

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.00
13:             73712              9233          00:00:00:00.04
14:            365596             45752          00:00:00:00.22
15:           2279184            285053          00:00:00:01.52
*/
void backTrack2_Recursive_BT_BM_SO_BOUND_BOUND2(int size,int mask,int row,int l,int d,int r){
  int bit;
  int bitmap=mask&~(l|d|r); /* 配置可能フィールド */
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  //if (row==size) {
  if (row==size-1) {
    //if(!bitmap){
    if(bitmap){
      if((bitmap&LASTMASK)==0){
        aBoard[row]=bitmap;
        symmetryOps_bm(size);
      }
    }
  }else{
    // 追加はじめ
    if(row<BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~SIDEMASK;
    }else if(row==BOUND2) {     	//【枝刈り】下部サイド枝刈り
      if((d&SIDEMASK)==0){ return; }
      if((d&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
    // 追加終わり
    while(bitmap) {
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
      backTrack2_Recursive_BT_BM_SO_BOUND_BOUND2(size,mask,row+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
void backTrack1_Recursive_BT_BM_SO_BOUND_BOUND2(int size,int mask,int row,int l,int d,int r){
  int bit;
  int bitmap=mask&~(l|d|r); /* 配置可能フィールド */
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  //if (row==size) {
  if (row==size-1) {
    //if(!bitmap){
    if(bitmap){
      aBoard[row]=bitmap;
      //symmetryOps_bm(size);
      C8++;
    }
  }else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bitmap) {
      bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //SO最も下位の１ビットを抽出
      backTrack1_Recursive_BT_BM_SO_BOUND_BOUND2(size,mask,row+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
void solve_nqueen_Recursive_BT_BM_SO_BOUND_BOUND2(int size,int mask) {
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    backTrack1_Recursive_BT_BM_SO_BOUND_BOUND2(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2_Recursive_BT_BM_SO_BOUND_BOUND2(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}




/**
  case 10 : 非再帰 非CUDA
  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
  4. 最上段のクイーンの位置による枝刈り BOUND 
  5. BOUNDの枝刈り

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.01
13:             73712              9233          00:00:00:00.05
14:            365596             45752          00:00:00:00.29
15:           2279184            285053          00:00:00:01.94
*/

void backTrack2_nonRecursive_BT_BM_SO_BOUND_BOUND2(int size,int mask,int row, int l, int d, int r){
  struct STACK stParam_2;
  for (int m=0;m<size;m++){ 
    stParam_2.param[m].Y=0;
    stParam_2.param[m].I=size;
    stParam_2.param[m].M=0;
    stParam_2.param[m].L=0;
    stParam_2.param[m].D=0;
    stParam_2.param[m].R=0;
    stParam_2.param[m].B=0;
  }
  stParam_2.current=0;
  int bend_2=0;
  int rflg_2=0;
  int bitmap;
  int bit;
  while(1){
   if(rflg_2==0){ 
    bitmap=mask&~(l|d|r); /* 配置可能フィールド */
   }
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    if (row==size-1&&rflg_2==0) {
      if(bitmap){
        if((bitmap&LASTMASK)==0){
          aBoard[row]=bitmap;
          symmetryOps_bm(size);
        }
      }
    }else{
    // 追加はじめ
        if(row<BOUND1&&rflg_2==0){             	//【枝刈り】上部サイド枝刈り
          bitmap&=~SIDEMASK;
        }else if(row==BOUND2&&rflg_2==0) {     	//【枝刈り】下部サイド枝刈り
          if((d&SIDEMASK)==0&&rflg_2==0){ 
            rflg_2=1;
          }
          if((d&SIDEMASK)!=SIDEMASK&&rflg_2==0){ bitmap&=SIDEMASK; }
        }
    // 追加終わり
      while(bitmap||rflg_2==1) {
        if(rflg_2==0){
          bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
          //backTrack2_nonRecursive_BT_BM_SO_BOUND_BOUND2(size,mask,row+1,(l|bit)<<1,d|bit,(r|bit)>>1);
          if(stParam_2.current<MAX){
            stParam_2.param[stParam_2.current].Y=row;
            stParam_2.param[stParam_2.current].I=size;
            stParam_2.param[stParam_2.current].M=mask;
            stParam_2.param[stParam_2.current].L=l;
            stParam_2.param[stParam_2.current].D=d;
            stParam_2.param[stParam_2.current].R=r;
            stParam_2.param[stParam_2.current].B=bitmap;
            (stParam_2.current)++;
          }
          row=row+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_2=1;
          break;
        }
        if(rflg_2==1){ 
          if(stParam_2.current>0){
            stParam_2.current--;
          }
          size=stParam_2.param[stParam_2.current].I;
          row=stParam_2.param[stParam_2.current].Y;
          mask=stParam_2.param[stParam_2.current].M;
          l=stParam_2.param[stParam_2.current].L;
          d=stParam_2.param[stParam_2.current].D;
          r=stParam_2.param[stParam_2.current].R;
          bitmap=stParam_2.param[stParam_2.current].B;
          rflg_2=0;
        }
      }
      if(bend_2==1 && rflg_2==0){
        bend_2=0;
        continue;
      }
    }
    if(row==1){
      break;
    }else{
      rflg_2=1;
    }
  } 
}
void backTrack1_nonRecursive_BT_BM_SO_BOUND_BOUND2(int size,int mask,int row, int l, int d, int r){
  struct STACK stParam_1;
  for (int m=0;m<size;m++){ 
    stParam_1.param[m].Y=0;
    stParam_1.param[m].I=size;
    stParam_1.param[m].M=0;
    stParam_1.param[m].L=0;
    stParam_1.param[m].D=0;
    stParam_1.param[m].R=0;
    stParam_1.param[m].B=0;
  }
  stParam_1.current=0;
  int bend_1=0;
  int rflg_1=0;
  int bit;
  int bitmap;
  while(1){
    if(rflg_1==0){
      bitmap=mask&~(l|d|r); /* 配置可能フィールド */
    }
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    if (row==size-1&&rflg_1==0) {
      if(bitmap){
        aBoard[row]=bitmap;
        C8++;
      }
    }else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      if(row<BOUND1 && rflg_1==0) {
        bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      while(bitmap||rflg_1==1) {
        if(rflg_1==0){
          bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //SO最も下位の１ビットを抽出
          //backTrack1_Recursive_BT_BM_SO_BOUND_BOUND2(size,mask,row+1,(l|bit)<<1,d|bit,(r|bit)>>1);
          if(stParam_1.current<MAX){
            stParam_1.param[stParam_1.current].Y=row;
            stParam_1.param[stParam_1.current].I=size;
            stParam_1.param[stParam_1.current].M=mask;
            stParam_1.param[stParam_1.current].L=l;
            stParam_1.param[stParam_1.current].D=d;
            stParam_1.param[stParam_1.current].R=r;
            stParam_1.param[stParam_1.current].B=bitmap;
            (stParam_1.current)++;
          }
          row=row+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_1=1;
          break;
        }
        if(rflg_1==1){ 
          if(stParam_1.current>0){
            stParam_1.current--;
          }
          size=stParam_1.param[stParam_1.current].I;
          row=stParam_1.param[stParam_1.current].Y;
          mask=stParam_1.param[stParam_1.current].M;
          l=stParam_1.param[stParam_1.current].L;
          d=stParam_1.param[stParam_1.current].D;
          r=stParam_1.param[stParam_1.current].R;
          bitmap=stParam_1.param[stParam_1.current].B;
          rflg_1=0;
        }
      }
      if(bend_1==1 && rflg_1==0){
        bend_1=0;
        continue;
      }
    } 
    if(row==2){
      break;
    }else{
      rflg_1=1;
    }
  } 
}
void solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2(int size,int mask){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  /* 最上段行のクイーンが角にある場合の探索 */
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    // 角にクイーンを配置 
    aBoard[1]=bit=(1<<BOUND1); 
    //２行目から探索
    backTrack1_nonRecursive_BT_BM_SO_BOUND_BOUND2(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1)); 
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2_nonRecursive_BT_BM_SO_BOUND_BOUND2(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
/**
  case 11 : 再帰 非CUDA 07_11相当
  07_11
  15:      2279184          285053            0.54

  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
  4. 最上段のクイーンの位置による枝刈り BOUND 
  5. BOUNDの枝刈り
  6. 最適化

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.00
13:             73712              9233          00:00:00:00.01
14:            365596             45752          00:00:00:00.08
15:           2279184            285053          00:00:00:00.53
*/
void symmetryOps_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(int size){
  int own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=size-1){ bit=1; you=size-1;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>size-1){ C2++; return; }
  }
  //180度回転
  if(aBoard[size-1]==ENDBIT){ own=1; you=size-1-1;
    while(own<=size-1){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>size-1){ C4++; return; }
  }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=size-1){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  C8++;
}
void backTrack2_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right);
	if(row==size-1){ 								// 【枝刈り】
		if(bitmap){
			if((bitmap&LASTMASK)==0){ 	//【枝刈り】 最下段枝刈り
				aBoard[row]=bitmap;
				symmetryOps_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(size);
			}
		}
	}else{
    if(row<BOUND1){             	//【枝刈り】上部サイド枝刈り
      bitmap&=~SIDEMASK;
    }else if(row==BOUND2) {     	//【枝刈り】下部サイド枝刈り
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
			backTrack2_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}

void backTrack1_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==size-1) {
    if(bitmap){
      aBoard[row]=bitmap;
      C8++;
    }
  }else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
			backTrack1_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}

void solve_nqueen_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(int size,int mask){
	int bit;
	TOPBIT=1<<(size-1);
	aBoard[0]=1;
	for(BOUND1=2;BOUND1<size-1;BOUND1++){
		aBoard[1]=bit=(1<<BOUND1);
		backTrack1_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
	}
	SIDEMASK=LASTMASK=(TOPBIT|1);
	ENDBIT=(TOPBIT>>1);
	for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
		aBoard[0]=bit=(1<<BOUND1);
		backTrack2_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(size,mask,1,bit<<1,bit,bit>>1);
		LASTMASK|=LASTMASK>>1|LASTMASK<<1;
		ENDBIT>>=1;
	}
}


/**
  case 12 : 非再帰 非CUDA
  1. バックトラック BT
  2. ビットマップ   BM
  3. 対象解除法     SO
  4. 最上段のクイーンの位置による枝刈り BOUND 
  5. BOUNDの枝刈り
  6. 最適化

 N:          Total        Unique                 dd:hh:mm:ss.ms
 4:                 2                 1          00:00:00:00.00
 5:                10                 2          00:00:00:00.00
 6:                 4                 1          00:00:00:00.00
 7:                40                 6          00:00:00:00.00
 8:                92                12          00:00:00:00.00
 9:               352                46          00:00:00:00.00
10:               724                92          00:00:00:00.00
11:              2680               341          00:00:00:00.00
12:             14200              1787          00:00:00:00.00
13:             73712              9233          00:00:00:00.02
14:            365596             45752          00:00:00:00.15
15:           2279184            285053          00:00:00:00.97
*/
void backTrack2_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT(int size,int mask,int row, int l, int d, int r){
  struct STACK stParam_2;
  for (int m=0;m<size;m++){ 
    stParam_2.param[m].Y=0;
    stParam_2.param[m].I=size;
    stParam_2.param[m].M=0;
    stParam_2.param[m].L=0;
    stParam_2.param[m].D=0;
    stParam_2.param[m].R=0;
    stParam_2.param[m].B=0;
  }
  stParam_2.current=0;
  int bend_2=0;
  int rflg_2=0;
  int bitmap;
  int bit;
  while(1){
   if(rflg_2==0){ 
    bitmap=mask&~(l|d|r); /* 配置可能フィールド */
   }
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    if (row==size-1&&rflg_2==0) {
      if(bitmap){
        if((bitmap&LASTMASK)==0){
          aBoard[row]=bitmap;
				  symmetryOps_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(size);
        }
      }
    }else{
    // 追加はじめ
        if(row<BOUND1&&rflg_2==0){             	//【枝刈り】上部サイド枝刈り
          bitmap&=~SIDEMASK;
        }else if(row==BOUND2&&rflg_2==0) {     	//【枝刈り】下部サイド枝刈り
          if((d&SIDEMASK)==0&&rflg_2==0){ 
            rflg_2=1;
          }
          if((d&SIDEMASK)!=SIDEMASK&&rflg_2==0){ bitmap&=SIDEMASK; }
        }
    // 追加終わり
      while(bitmap||rflg_2==1) {
        if(rflg_2==0){
          bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
          if(stParam_2.current<MAX){
            stParam_2.param[stParam_2.current].Y=row;
            stParam_2.param[stParam_2.current].I=size;
            stParam_2.param[stParam_2.current].M=mask;
            stParam_2.param[stParam_2.current].L=l;
            stParam_2.param[stParam_2.current].D=d;
            stParam_2.param[stParam_2.current].R=r;
            stParam_2.param[stParam_2.current].B=bitmap;
            (stParam_2.current)++;
          }
          row=row+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_2=1;
          break;
        }
        if(rflg_2==1){ 
          if(stParam_2.current>0){
            stParam_2.current--;
          }
          size=stParam_2.param[stParam_2.current].I;
          row=stParam_2.param[stParam_2.current].Y;
          mask=stParam_2.param[stParam_2.current].M;
          l=stParam_2.param[stParam_2.current].L;
          d=stParam_2.param[stParam_2.current].D;
          r=stParam_2.param[stParam_2.current].R;
          bitmap=stParam_2.param[stParam_2.current].B;
          rflg_2=0;
        }
      }
      if(bend_2==1 && rflg_2==0){
        bend_2=0;
        continue;
      }
    }
    if(row==1){
      break;
    }else{
      rflg_2=1;
    }
  } 
}
void backTrack1_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT(int size,int mask,int row, int l, int d, int r){
  struct STACK stParam_1;
  for (int m=0;m<size;m++){ 
    stParam_1.param[m].Y=0;
    stParam_1.param[m].I=size;
    stParam_1.param[m].M=0;
    stParam_1.param[m].L=0;
    stParam_1.param[m].D=0;
    stParam_1.param[m].R=0;
    stParam_1.param[m].B=0;
  }
  stParam_1.current=0;
  int bend_1=0;
  int rflg_1=0;
  int bit;
  int bitmap;
  while(1){
    if(rflg_1==0){
      bitmap=mask&~(l|d|r); /* 配置可能フィールド */
    }
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
    if (row==size-1&&rflg_1==0) {
      if(bitmap){
        aBoard[row]=bitmap;
        C8++;
      }
    }else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      if(row<BOUND1 && rflg_1==0) {
        bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      while(bitmap||rflg_1==1) {
        if(rflg_1==0){
          bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //SO最も下位の１ビットを抽出
          if(stParam_1.current<MAX){
            stParam_1.param[stParam_1.current].Y=row;
            stParam_1.param[stParam_1.current].I=size;
            stParam_1.param[stParam_1.current].M=mask;
            stParam_1.param[stParam_1.current].L=l;
            stParam_1.param[stParam_1.current].D=d;
            stParam_1.param[stParam_1.current].R=r;
            stParam_1.param[stParam_1.current].B=bitmap;
            (stParam_1.current)++;
          }
          row=row+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_1=1;
          break;
        }
        if(rflg_1==1){ 
          if(stParam_1.current>0){
            stParam_1.current--;
          }
          size=stParam_1.param[stParam_1.current].I;
          row=stParam_1.param[stParam_1.current].Y;
          mask=stParam_1.param[stParam_1.current].M;
          l=stParam_1.param[stParam_1.current].L;
          d=stParam_1.param[stParam_1.current].D;
          r=stParam_1.param[stParam_1.current].R;
          bitmap=stParam_1.param[stParam_1.current].B;
          rflg_1=0;
        }
      }
      if(bend_1==1 && rflg_1==0){
        bend_1=0;
        continue;
      }
    } 
    if(row==2){
      break;
    }else{
      rflg_1=1;
    }
  } 
}
void solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT(int size,int mask){
  int bit;
  TOPBIT=1<<(size-1);
  aBoard[0]=1;
  /* 最上段行のクイーンが角にある場合の探索 */
  for(BOUND1=2;BOUND1<size-1;BOUND1++){
    // 角にクイーンを配置 
    aBoard[1]=bit=(1<<BOUND1); 
    //２行目から探索
    backTrack1_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1)); 
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT(size,mask,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
/** #################################################################

  nVidia CUDA ブロック

#####################################################################*/
/** 
  CUDA 非再帰 CPUイテレータから複数の初期条件を受け取り、カウント
  1. バックトラック backTrack
  2. ビットマップ   bitmap
14:            365596                 0          00:00:00:00.08
15:           2279184                 0          00:00:00:00.49
*/
__global__ void solve_nqueen_cuda_kernel_bt_bm(
  int n,int mark,
  unsigned int* totalDown,unsigned int* totalLeft,unsigned int* totalRight,
  unsigned int* results,int totalCond){
  const int tid=threadIdx.x,bid=blockIdx.x,idx=bid*blockDim.x+tid;
  __shared__ unsigned int down[THREAD_NUM][10],left[THREAD_NUM][10],right[THREAD_NUM][10],
                          bitmap[THREAD_NUM][10],sum[THREAD_NUM];
  const unsigned int mask=(1<<n)-1;int total=0,i=0;unsigned int bit;
  if(idx<totalCond){
    down[tid][i]=totalDown[idx];
    left[tid][i]=totalLeft[idx];
    right[tid][i]=totalRight[idx];
    bitmap[tid][i]=down[tid][i]|left[tid][i]|right[tid][i];
    while(i>=0){
      if((bitmap[tid][i]&mask)==mask){i--;}
      else{
        bit=(bitmap[tid][i]+1)&~bitmap[tid][i];
        bitmap[tid][i]|=bit;
        if((bit&mask)!=0){
          if(i+1==mark){total++;i--;}
          else{
            down[tid][i+1]=down[tid][i]|bit;
            left[tid][i+1]=(left[tid][i]|bit)<<1;
            right[tid][i+1]=(right[tid][i]|bit)>>1;
            bitmap[tid][i+1]=(down[tid][i+1]|left[tid][i+1]|right[tid][i+1]);
            i++;
          }
        }else{i--;}
      }
    }
    sum[tid]=total;
  }else{sum[tid]=0;} 
  __syncthreads();if(tid<64&&tid+64<THREAD_NUM){sum[tid]+=sum[tid+64];} 
  __syncthreads();if(tid<32){sum[tid]+=sum[tid+32];} 
  __syncthreads();if(tid<16){sum[tid]+=sum[tid+16];} 
  __syncthreads();if(tid<8){sum[tid]+=sum[tid+8];} 
  __syncthreads();if(tid<4){sum[tid]+=sum[tid+4];} 
  __syncthreads();if(tid<2){sum[tid]+=sum[tid+2];} 
  __syncthreads();if(tid<1){sum[tid]+=sum[tid+1];} 
  __syncthreads();if(tid==0){results[bid]=sum[0];}
}
long long solve_nqueen_cuda(int n,int steps) {
  unsigned int down[32];unsigned int left[32];unsigned int right[32];
  unsigned int m[32];unsigned int bit;
  if(n<=0||n>32){return 0;}
  unsigned int* totalDown=new unsigned int[steps];
  unsigned int* totalLeft=new unsigned int[steps];
  unsigned int* totalRight=new unsigned int[steps];
  unsigned int* results=new unsigned int[steps];
  unsigned int* downCuda;unsigned int* leftCuda;unsigned int* rightCuda;
  unsigned int* resultsCuda;
  cudaMalloc((void**) &downCuda,sizeof(int)*steps);
  cudaMalloc((void**) &leftCuda,sizeof(int)*steps);
  cudaMalloc((void**) &rightCuda,sizeof(int)*steps);
  cudaMalloc((void**) &resultsCuda,sizeof(int)*steps/THREAD_NUM);
  const unsigned int mask=(1<<n)-1;
  const unsigned int mark=n>11?n-10:2;
  long long total=0;int totalCond=0;
  int i=0,j;down[0]=0;left[0]=0;right[0]=0;m[0]=0;bool computed=false;
  for(j=0;j<n/2;j++){
    bit=(1<<j);m[0]|=bit;
    down[1]=bit;left[1]=bit<<1;right[1]=bit>>1;
    m[1]=(down[1]|left[1]|right[1]);
    i=1;
    while(i>0){
      if((m[i]&mask)==mask){i--;}
      else{
        bit=(m[i]+1)&~m[i];m[i]|=bit;
        if((bit&mask)!=0){
          down[i+1]=down[i]|bit;left[i+1]=(left[i]|bit)<<1;right[i+1]=(right[i]|bit)>>1;
          m[i+1]=(down[i+1]|left[i+1]|right[i+1]);
          i++;
          if(i==mark){
            totalDown[totalCond]=down[i];totalLeft[totalCond]=left[i];totalRight[totalCond]=right[i];
            totalCond++;
            if(totalCond==steps){
              if(computed){
                cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
                computed=false;
              }
              cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
              computed=true;totalCond=0;
            }
            i--;
          }
        }else{i --;}
      }
    }
  }
  if(computed){
    cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
    computed=false;
  }
  cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
  /** backTrack+bitmap*/
  solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
  cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
  for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}	
  total*=2;
  if(n%2==1){
    computed=false;totalCond=0;bit=(1<<(n-1)/2);m[0]|=bit;
    down[1]=bit;left[1]=bit<<1;right[1]=bit>>1;
    m[1]=(down[1]|left[1]|right[1]);
    i=1;
    while(i>0){
      if((m[i]&mask)==mask){i--;}
      else{
        bit=(m[i]+1)&~m[i];m[i]|=bit;
        if((bit&mask)!=0){
          down[i+1]=down[i]|bit;left[i+1]=(left[i]|bit)<<1;right[i+1]=(right[i]|bit)>>1;
          m[i+1]=(down[i+1]|left[i+1]|right[i+1]);
          i++;
          if(i==mark){
            totalDown[totalCond]=down[i];totalLeft[totalCond]=left[i];totalRight[totalCond]=right[i];
            totalCond++;
            if(totalCond==steps){
              if(computed){
                cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
                for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
                computed=false;
              }
              cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
              /** backTrack+bitmap*/
              solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
              computed=true;totalCond=0;
            }
            i--;
          }
        }else{i --;}
      }
    }
    if(computed){
      cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
      for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
      computed=false;
    }
    cudaMemcpy(downCuda,totalDown,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(leftCuda,totalLeft,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    cudaMemcpy(rightCuda,totalRight,sizeof(int)*totalCond,cudaMemcpyHostToDevice);
    /** backTrack+bitmap*/
    solve_nqueen_cuda_kernel_bt_bm<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n-mark,downCuda,leftCuda,rightCuda,resultsCuda,totalCond);
    cudaMemcpy(results,resultsCuda,sizeof(int)*steps/THREAD_NUM,cudaMemcpyDeviceToHost);
    for(int j=0;j<steps/THREAD_NUM;j++){total+=results[j];}
  }
  cudaFree(downCuda);cudaFree(leftCuda);cudaFree(rightCuda);cudaFree(resultsCuda);
  delete[] totalDown;delete[] totalLeft;delete[] totalRight;delete[] results;
  return total;
}
/** CUDA 初期化 **/
bool InitCUDA(){
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  int i;
  for(i=0;i<count;i++){
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
void execCPU(int procNo){
  int min=4;int targetN=17;
  int msk;
  struct timeval t0;struct timeval t1;int ss;int ms;int dd;
  printf("\n%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=targetN;i++){
    Total=Unique=C2=C4=C8=0;
    gettimeofday(&t0,NULL);   // 計測開始
    switch (procNo){
      case 1:
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        solve_nqueen_Recursive_BT(0,i);
        break;
      case 2:
        for(int j=0;j<i;j++){ aBoard[j]=-1; } //aBoardを初期化
        solve_nqueen_nonRecursive_BT(0,i);
        break;
      case 3:
        Total=solve_nqueen_Recursive_BT_BM(i,0,0,0);       
        break;
      case 4: 
        Total=solve_nqueen_nonRecursive_BT_BM(i);    
        break;
      case 5: 
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        solve_nqueen_Recursive_BT_BM_SO(i,msk,0,0,0,0);
        Total=getTotal();
        Unique=getUnique();
        break;
      case 6: 
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        Total=0;Unique=0;C2=0;C4=0;C8=0;
        solve_nqueen_nonRecursive_BT_BM_SO(i,msk,0,0,0,0);
        Total=getTotal();
        Unique=getUnique();
        break;
      case 7: 
        //Total=solve_nqueen_Recursive_BT_BM_SO_BOUND(i); 
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        Total=0;Unique=0;C2=0;C4=0;C8=0;
        solve_nqueen_Recursive_BT_BM_SO_BOUND(i,msk); 
        Total=getTotal();
        Unique=getUnique();
        break;
      case 8: 
        //Unique=solve_nqueen_nonRecursive_BT_BM_SO_BOUND(i); 
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        Total=0;Unique=0;C2=0;C4=0;C8=0;
        for(int BOUND1=0,BOUND2=i-1;BOUND1<i;BOUND1++,BOUND2--){
          solve_nqueen_nonRecursive_BT_BM_SO_BOUND(i,BOUND1,BOUND2,msk); 
        }
        Total=getTotal();
        Unique=getUnique();
        break;
      case 9:
        /* solve_nqueen_Recursive_BT_BM_SO_BOUND_BOUND2     */
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        Total=0;Unique=0;C2=0;C4=0;C8=0;
        solve_nqueen_Recursive_BT_BM_SO_BOUND_BOUND2(i,msk); 
        Total=getTotal();
        Unique=getUnique();
        break ;
      case 10:
        /* solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2     */
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        Total=0;Unique=0;C2=0;C4=0;C8=0;
        solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2(i,msk); 
        Total=getTotal();
        Unique=getUnique();
        break ;
      case 11:
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        Total=0;Unique=0;C2=0;C4=0;C8=0;
        solve_nqueen_Recursive_BT_BM_SO_BOUND_BOUND2_OPT(i,msk);
        Total=getTotal();
        Unique=getUnique();
      break ;
      case 12:
        for(int j=0;j<i;j++){ aBoard[j]=j; } //aBoardを初期化
        msk=(1<<i)-1; // 初期化
        Total=0;Unique=0;C2=0;C4=0;C8=0;
        solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT(i,msk);
        Total=getTotal();
        Unique=getUnique();
        break ;
      default: break;
    } 
    gettimeofday(&t1,NULL);   // 計測終了
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
    printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", i,Total,Unique,dd,hh,mm,ss,ms);
  }
}
int main(int argc,char** argv) {
  bool cpu=true,gpu=true;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){gpu=false;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){cpu=false;}
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g] n steps\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -g: GPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  /** CPU */
  if(cpu){
    printf("\n\n1. 再帰＋バックトラック(BT)");
    execCPU(1);  // solve_nqueen_Recursive_BT     
    printf("\n\n2. 非再帰＋バックトラック(BT)");
    execCPU(2);  // solve_nqueen_nonRecursive_BT     
    printf("\n\n3. 再帰＋バックトラック(BT)＋ビットマップ(BM)");
    execCPU(3);  // solve_nqueen_Recursive_BT_BM  
    printf("\n\n4. 非再帰＋バックトラック(BT)＋ビットマップ(BM)");
    execCPU(4);  // solve_nqueen_nonRecursive_BT_BM  
    printf("\n\n5. 再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)");
    execCPU(5);  // solve_nqueen_Recursive_BT_BM_SO     
    printf("\n\n6. 非再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)");
    execCPU(6);  // solve_nqueen_nonRecursive_BT_BM_SO     
    printf("\n\n7. 再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)＋枝刈り(BOUND)");
    execCPU(7);  // solve_nqueen_Recursive_BT_BM_SO_BOUND     
    printf("\n\n8. 非再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)＋枝刈り(BOUND)");
    execCPU(8);  // solve_nqueen_nonRecursive_BT_BM_SO_BOUND     
    printf("\n\n9. 再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)＋枝刈り(BOUND)＋BOUNDの枝刈り");
    execCPU(9);  // solve_nqueen_Recursive_BT_BM_SO_BOUND_BOUND2     
    printf("\n\n10. 非再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)＋枝刈り(BOUND)＋BOUNDの枝刈り");
    execCPU(10);  // solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2     
    printf("\n\n11. 再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)＋枝刈り(BOUND)＋BOUNDの枝刈り＋最適化");
    execCPU(11);  // solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT     
    printf("\n\n12. 非再帰＋バックトラック(BT)＋ビットマップ(BM)＋対象解除法(SO)＋枝刈り(BOUND)＋BOUNDの枝刈り＋最適化");
    execCPU(12);  // solve_nqueen_nonRecursive_BT_BM_SO_BOUND_BOUND2_OPT     
  }
  /** GPU */
  if(gpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=17;
    struct timeval t0;struct timeval t1;int ss;int ms;int dd;
    printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      Total=solve_nqueen_cuda(i,steps);
      gettimeofday(&t1,NULL);   // 計測終了
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
      printf("%2d:%18ld%18ld%12.2d:%02d:%02d:%02d.%02d\n", i,Total,Unique,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}


