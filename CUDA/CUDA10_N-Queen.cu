/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ nvcc CUDA01_N-Queen.cu -o CUDA01_N-Queen

 実行
 $ ./CUDA01_N-Queen

 1. ブルートフォース　力任せ探索

 　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解とな
 るかをチェックする方法

   (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
   けであって8王妃問題を解いているわけではありません

 実行結果
 :
 :
 16777209: 7 7 7 7 7 7 7 0
 16777210: 7 7 7 7 7 7 7 1
 16777211: 7 7 7 7 7 7 7 2
 16777212: 7 7 7 7 7 7 7 3
 16777213: 7 7 7 7 7 7 7 4
 16777214: 7 7 7 7 7 7 7 5
 16777215: 7 7 7 7 7 7 7 6
 16777216: 7 7 7 7 7 7 7 7

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define THREAD_NUM		96
#define MAX 27
//
long Total=0 ;        //合計解
long Unique=0;
int down[2*MAX-1]; //down:flagA 縦 配置フラグ　
int left[2*MAX-1];  //left:flagB 斜め配置フラグ　
int right[2*MAX-1];  //right:flagC 斜め配置フラグ　
int aBoard[MAX];
int aT[MAX];
int aS[MAX];
int bit;
int COUNT2,COUNT4,COUNT8;
int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
//
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
//
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
//main()以外のメソッドはここに一覧表記させます
void TimeFormat(clock_t utime,char *form);
void rotate_bitmap(int bf[],int af[],int si);
void vMirror_bitmap(int bf[],int af[],int si);
int intncmp(int lt[],int rt[],int n);
void dtob(int score,int si);
int rh(int a,int sz);
long getUnique();
long getTotal();
void symmetryOps_bitmap(int si);
void backTrack2(int si,int mask,int y,int l,int d,int r);
void backTrack1(int si,int mask,int y,int l,int d,int r);
void NQueen(int size,int mask);
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char *form){
	int dd,hh,mm;
	float ftime,ss;
	ftime=(float)utime/CLOCKS_PER_SEC;
	mm=(int)ftime/60;
	ss=ftime-(int)(mm*60);
	dd=mm/(24*60);
	mm=mm%(24*60);
	hh=mm/60;
	mm=mm%60;
	if(dd)
	sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
	else if(hh)
	sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
	else if(mm)
	sprintf(form,"        %2d:%05.2f",mm,ss);
	else
	sprintf(form,"           %5.2f",ss);
}
//
void dtob(int score,int size){
	int bit=1;
	char c[size];
	for(int i=0;i<size;i++){
		if(score&bit){
			c[i]='1';
		}else{
			c[i]='0';
		}
		bit<<=1;
	}
	for(int i=size-1;i>=0;i--){
		putchar(c[i]);
	}
	printf("\n");
}
//
int rh(int a,int size){
	int tmp=0;
	for(int i=0;i<=size;i++){
		if(a&(1<<i)){
			return tmp|=(1<<(size-i));
		}
	}
	return tmp;
}
//
void vMirror_bitmap(int bf[],int af[],int size){
	int score;
	for(int i=0;i<size;i++){
		score=bf[i];
		af[i]=rh(score,size-1);
	}
}
//
void rotate_bitmap(int bf[],int af[],int size){
	int t;
	for(int i=0;i<size;i++){
		t=0;
		for(int j=0;j<size;j++){
			t|=((bf[j]>>i)&1)<<(size-j-1); // x[j] の i ビット目を
		}
		af[i]=t;                        // y[i] の j ビット目にする
	}
}
//
int intncmp(int lt[],int rt[],int n){
	int rtn=0;
	for(int k=0;k<n;k++){
		rtn=lt[k]-rt[k];
		if(rtn!=0){
			break;
		}
	}
	return rtn;
}
//
long getUnique(){
	return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
	return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps_bitmap(int size){
	int nEquiv;
	// 回転・反転・対称チェックのためにboard配列をコピー
	for(int i=0;i<size;i++){
		aT[i]=aBoard[i];
	}
	rotate_bitmap(aT,aS,size);    //時計回りに90度回転
	int k=intncmp(aBoard,aS,size);
	if(k>0) return;
	if(k==0){
		nEquiv=2;
	}else{
		rotate_bitmap(aS,aT,size);  //時計回りに180度回転
		k=intncmp(aBoard,aT,size);
		if(k>0) return;
		if(k==0){
			nEquiv=4;
		}else{
			rotate_bitmap(aT,aS,size);  //時計回りに270度回転
			k=intncmp(aBoard,aS,size);
			if(k>0){
				return;
			}
			nEquiv=8;
		}
	}
	// 回転・反転・対称チェックのためにboard配列をコピー
	for(int i=0;i<size;i++){
		aS[i]=aBoard[i];
	}
	vMirror_bitmap(aS,aT,size);   //垂直反転
	k=intncmp(aBoard,aT,size);
	if(k>0){
		return;
	}
	if(nEquiv>2){             //-90度回転 対角鏡と同等
		rotate_bitmap(aT,aS,size);
		k=intncmp(aBoard,aS,size);
		if(k>0){
			return;
		}
		if(nEquiv>4){           //-180度回転 水平鏡像と同等
			rotate_bitmap(aS,aT,size);
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return;
			}       //-270度回転 反対角鏡と同等
			rotate_bitmap(aT,aS,size);
			k=intncmp(aBoard,aS,size);
			if(k>0){
				return;
			}
		}
	}
	if(nEquiv==2){
		COUNT2++;
	}
	if(nEquiv==4){
		COUNT4++;
	}
	if(nEquiv==8){
		COUNT8++;
	}
}
//
void backTrack2(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right); /* 配置可能フィールド */
	if(row==size){
		if(!bitmap){
			aBoard[row]=bitmap;
			symmetryOps_bitmap(size);
		}
	}else{
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
			backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}
//
void backTrack1(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right); /* 配置可能フィールド */
	if(row==size){
		if(!bitmap){
			aBoard[row]=bitmap;
			symmetryOps_bitmap(size);
		}
	}else{
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap); //最も下位の１ビットを抽出
			backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}
//
void NQueen(int size,int mask){
	int bit;
	TOPBIT=1<<(size-1);
	aBoard[0]=1;
	for(BOUND1=2;BOUND1<size-1;BOUND1++){
		aBoard[1]=bit=(1<<BOUND1);
		backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
	}
	SIDEMASK=LASTMASK=(TOPBIT|1);
	ENDBIT=(TOPBIT>>1);
	for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
		aBoard[0]=bit=(1<<BOUND1);
		backTrack2(size,mask,1,bit<<1,bit,bit>>1);
		LASTMASK|=LASTMASK>>1|LASTMASK<<1;
		ENDBIT>>=1;
	}
}
//メインメソッド
int main(int argc,char** argv) {
  bool cpu=true,cpur=true,gpu=true;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){gpu=false;cpur=false;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpu=false;gpu=false;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){cpu=false;cpur=false;}
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r] n steps\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  /** CPU */
  if(cpu){
    printf("\n\n１０．クイーンの位置による分岐BOUND1,2");
  }
  /** CPUR */
  if(cpur){
    printf("\n\n１０．クイーンの位置による分岐BOUND1,2");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    int min=4;            //Nの最小値（スタートの値）を格納
    int mask=0;
    int max=17;
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    for(int i=min;i<=max;i++){
      COUNT2=COUNT4=COUNT8=0;
      mask=(1<<i)-1;
      for(int j=0;j<i;j++){ aBoard[j]=j; } //版を初期化
        st=clock();         //計測開始
	NQueen(i,mask);
        TimeFormat(clock()-st,t); //計測終了
        printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t); //出力
    }
    return 0;
  }
  /** GPU */
  if(gpu){
    if(!InitCUDA()){return 0;}
    int min=4;int targetN=18;
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
