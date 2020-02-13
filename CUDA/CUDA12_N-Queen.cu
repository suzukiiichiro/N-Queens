/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ nvcc CUDA12_N-Queen.cu -o CUDA12_N-Queen

 実行
 $ ./CUDA12_N-Queen (-c|-r|-g)
                    -c:cpu -r cpu再帰 -g GPU

 １２．対称解除法の最適化

 実行結果

１２．CPUR 再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.02
14:       365596           45752            0.08
15:      2279184          285053            0.52
16:     14772512         1846955            3.35
17:     95815104        11977939           23.19

１２．CPU 非再帰 対称解除法の最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.02
14:       365596           45752            0.11
15:      2279184          285053            0.65
16:     14772512         1846955            4.26
17:     95815104        11977939           29.41
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
//変数宣言
long Total=0 ;      //合計解
long Unique=0;
int aBoard[MAX];
int COUNT2,COUNT4,COUNT8;
int BOUND1,BOUND2,TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
//関数宣言
void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void symmetryOps(int si);
void backTrack2_NR(int si,int mask,int y,int l,int d,int r);
void backTrack1_NR(int si,int mask,int y,int l,int d,int r);
void NQueen(int size,int mask);
void backTrack2(int si,int mask,int y,int l,int d,int r);
void backTrack1(int si,int mask,int y,int l,int d,int r);
void NQueenR(int size,int mask);
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
long getUnique(){
	return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
	return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps(int si){
  int own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ COUNT2++; return; }
  }
  //180度回転
  if(aBoard[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ COUNT4++; return; }
  }
  //270度回転
  if(aBoard[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  COUNT8++;
}
//CPU 非再帰版 backTrack2
void backTrack2_NR(int size,int mask,int row,int left,int down,int right){
	int bitmap,bit;
	int b[100], *p=b;
  int sizeE=size-1;
	mais1:bitmap=mask&~(left|down|right);
  // 【枝刈り】
	//if(row==size){
	if(row==sizeE){
		//if(!bitmap){
		if(bitmap){
      //【枝刈り】 最下段枝刈り
			if((bitmap&LASTMASK)==0){
				aBoard[row]=bitmap;
				symmetryOps(size);
			}
		}
	}else{
    //【枝刈り】上部サイド枝刈り
		if(row<BOUND1){
			bitmap&=~SIDEMASK;
    //【枝刈り】下部サイド枝刈り
		}else if(row==BOUND2){
			if(!(down&SIDEMASK))
				goto volta;
			if((down&SIDEMASK)!=SIDEMASK)
				bitmap&=SIDEMASK;
		}
		if(bitmap){
			outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
			if(bitmap){
				*p++=left;
				*p++=down;
				*p++=right;
			}
			*p++=bitmap;
			row++;
			left=(left|bit)<<1;
			down=down|bit;
			right=(right|bit)>>1;
			goto mais1;
			//Backtrack2(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
			volta:if(p<=b)
				return;
			row--;
			bitmap=*--p;
			if(bitmap){
				right=*--p;
				down=*--p;
				left=*--p;
				goto outro;
			}else{
				goto volta;
			}
		}
	}
	goto volta;
}
//CPU 非再帰版 backTrack
void backTrack1_NR(int size,int mask,int row,int left,int down,int right){
	int bitmap,bit;
	int b[100], *p=b;
  int sizeE=size-1;
	b1mais1:bitmap=mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
	//if(row==size){
	if(row==sizeE){
		//if(!bitmap){
		if(bitmap){
			aBoard[row]=bitmap;
			//symmetryOps_bitmap(size);
			COUNT8++;
		}
	}else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
		if(bitmap){
			b1outro:bitmap^=aBoard[row]=bit=-bitmap&bitmap;
			if(bitmap){
				*p++=left;
				*p++=down;
				*p++=right;
			}
			*p++=bitmap;
			row++;
			left=(left|bit)<<1;
			down=down|bit;
			right=(right|bit)>>1;
			goto b1mais1;
			//Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
			b1volta:if(p<=b)
				return;
			row--;
			bitmap=*--p;
			if(bitmap){
				right=*--p;
				down=*--p;
				left=*--p;
				goto b1outro;
			}else{
				goto b1volta;
			}
		}
	}
	goto b1volta;
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask){
	int bit;
	TOPBIT=1<<(size-1);
	aBoard[0]=1;
	for(BOUND1=2;BOUND1<size-1;BOUND1++){
		aBoard[1]=bit=(1<<BOUND1);
		//backTrack1(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
		backTrack1_NR(size,mask,2,(2|bit)<<1,(1|bit),(bit>>1));
	}
	SIDEMASK=LASTMASK=(TOPBIT|1);
	ENDBIT=(TOPBIT>>1);
	for(BOUND1=1,BOUND2=size-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
		aBoard[0]=bit=(1<<BOUND1);
		//backTrack1(size,mask,1,bit<<1,bit,bit>>1);
		backTrack2_NR(size,mask,1,bit<<1,bit,bit>>1);
		LASTMASK|=LASTMASK>>1|LASTMASK<<1;
		ENDBIT>>=1;
	}
}
//
void backTrack2(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right);
  // 【枝刈り】
	if(row==size-1){ 								
		if(bitmap){
      //【枝刈り】 最下段枝刈り
			if((bitmap&LASTMASK)==0){ 	
				aBoard[row]=bitmap;
				symmetryOps(size);
			}
		}
	}else{
    //【枝刈り】上部サイド枝刈り
    if(row<BOUND1){
      bitmap&=~SIDEMASK;
    //【枝刈り】下部サイド枝刈り
    }else if(row==BOUND2) {
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
			backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}
//
void backTrack1(int size,int mask,int row,int left,int down,int right){
	int bit;
	int bitmap=mask&~(left|down|right);
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(row==size-1) {
    if(bitmap){
      aBoard[row]=bitmap;
      COUNT8++;
    }
  }else{
		//【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
		// ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    if(row<BOUND1) {
      bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
		while(bitmap){
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
			backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}
//
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,int mask){
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
  bool cpu=false,cpur=false,gpu=false;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else
      cpur=true;
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
  if(cpu){
    printf("\n\n１２．CPU 非再帰 対称解除法の最適化\n");
  }else if(cpur){
    printf("\n\n１２．CPUR 再帰 対称解除法の最適化\n");
  }else if(gpu){
    printf("\n\n１２．GPU 非再帰 対称解除法の最適化\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
		clock_t st;           //速度計測用
		char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=18;
    int mask;
    for(int i=min;i<=targetN;i++){
      //TOTAL=0; UNIQUE=0;
      COUNT2=COUNT4=COUNT8=0;
      mask=(1<<i)-1;
      st=clock();
      if(cpu){
        //非再帰は-1で初期化
        for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
        NQueen(i,mask);
      }
      if(cpur){
        //再帰は0で初期化
        //for(int j=0;j<=targetN;j++){ aBoard[j]=0; } 
        for(int j=0;j<=targetN;j++){ aBoard[j]=j; } 

        NQueenR(i,mask);
      }
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
    }
  }
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
