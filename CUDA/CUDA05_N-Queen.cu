/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ nvcc CUDA05_N-Queen.cu -o CUDA05_N-Queen

 実行
 $ ./CUDA05_N-Queen (-c|-r|-g)
                    -c:cpu -r cpu再帰 -g GPU

 * ５．バックトラック＋対称解除法＋枝刈りと最適化

 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。

実行結果

５．CPUR 再帰 バックトラック＋対称解除法＋枝刈りと最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1787            0.03
13:        73712            9233            0.17
14:       365596           45752            0.94
15:      2279184          285053            6.39
16:     14772512         1846955           40.57
17:     95815104        11977939         5:05.43

５．CPU 非再帰 バックトラック＋対称解除法＋枝刈りと最適化
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1787            0.04
13:        73712            9233            0.26
14:       365596           45752            1.46
15:      2279184          285053           10.02
16:     14772512         1846955         1:04.29
17:     95815104        11977939         8:12.65

５．GPU 再帰 バックトラック＋対称解除法＋枝刈りと最適化

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
long TOTAL=0;
long UNIQUE=0;
int aBoard[MAX];
int fA[2*MAX-1];	//縦列にクイーンを一つだけ配置
int fB[2*MAX-1];	//斜め列にクイーンを一つだけ配置
int fC[2*MAX-1];	//斜め列にクイーンを一つだけ配置
int aT[MAX];       //aT:aTrial[]
int aS[MAX];       //aS:aScrath[]
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
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);
int symmetryOps(int si);
void NQueen(int row,int size);
void NQueenR(int row,int size);
//
void rotate(int chk[],int scr[],int n,int neg){
	int k=neg ? 0 : n-1;
	int incr=(neg ? +1 : -1);
	for(int j=0;j<n;k+=incr){
		scr[j++]=chk[k];
	}
	k=neg ? n-1 : 0;
	for(int j=0;j<n;k-=incr){
		chk[scr[j++]]=k;
	}
}
//
void vMirror(int chk[],int n){
	for(int j=0;j<n;j++){
		chk[j]=(n-1)-chk[j];
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
int symmetryOps(int size){
	int nEquiv;
	// 回転・反転・対称チェックのためにboard配列をコピー
	for(int i=0;i<size;i++){
		aT[i]=aBoard[i];
	}
  //時計回りに90度回転
	rotate(aT,aS,size,0);       
	int k=intncmp(aBoard,aT,size);
	if(k>0) return 0;
	if(k==0){
		nEquiv=1;
	}else{
    //時計回りに180度回転
		rotate(aT,aS,size,0);     
		k=intncmp(aBoard,aT,size);
		if(k>0) return 0;
		if(k==0){
			nEquiv=2;
		}else{
      //時計回りに270度回転
			rotate(aT,aS,size,0);   
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return 0;
			}
			nEquiv=4;
		}
	}
	// 回転・反転・対称チェックのためにboard配列をコピー
	for(int i=0;i<size;i++){
		aT[i]=aBoard[i];
	}
  //垂直反転
	vMirror(aT,size);           
	k=intncmp(aBoard,aT,size);
	if(k>0){
		return 0;
	}
  //-90度回転 対角鏡と同等
	if(nEquiv>1){             
		rotate(aT,aS,size,1);
		k=intncmp(aBoard,aT,size);
		if(k>0){
			return 0;
		}
    //-180度回転 水平鏡像と同等
		if(nEquiv>2){           
			rotate(aT,aS,size,1);
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return 0;
			}  //-270度回転 反対角鏡と同等
			rotate(aT,aS,size,1);
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return 0;
			}
		}
	}
	return nEquiv*2;
}
// デバッグ用出力
int COUNT=0;
void print(int size){
	printf("%d: ",++COUNT);
	for(int j=0;j<size;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
	printf("\n");
	printf("\n");

}
void _NQueen(int row,int size){
  bool matched;
  while(row>=0){
    matched=false;
    //search begins at the position previously visited
		int lim=(row!=0)?size:(size+1)/2;
    //printf("aBoard[row]:%d\n",aBoard[row]);//-1
    for(int i=aBoard[row]+1;i<lim;i++){
      //the first matched position
      //if(down[i]==0&&left[row+(size-1)-i]==0&&right[row+i]==0){
      if(left[row+(size-1)-i]==0&&right[row+i]==0){
        //clear original record 
        if(aBoard[row]>=0){
          //down[aBoard[row]]=left[row+(size-1)-aBoard[row]]=right[row+aBoard[row]]=0;
          //left[row+(size-1)-i]=right[row+i]=0;
          left[row+(size-1)-aBoard[row]]=right[row+aBoard[row]]=0;

          /** 
            ここの交換が謎
          */
          //交換 
          /* int tmp=aBoard[row];      */
          /* for(int i=row;i<lim;i++){ */
          /*   aBoard[i]=aBoard[i+1];  */
          /* }                         */
          /* aBoard[size-1]=tmp;       */
        }

        aBoard[row]=i; // 交換するときはコメントアウト

        //交換
        /* int tmp=i;       */
        /* i=aBoard[row];   */
        /* aBoard[row]=tmp; */

        //down[i]=left[row+(size-1)-i]=right[row+i]=1;
        //left[row+(size-1)-i]=right[row+i]=1;
        left[row+(size-1)-aBoard[row]]=right[row+aBoard[row]]=1;
        matched=true;
        break;
      }
    }
    if(matched){
      //next aBoard
      row++;
      if(row==size-1){
        if(aBoard[row]!=-1){
          //if(down[aBoard[row]]==1||left[row-aBoard[row]+(size-1)]==1||right[row+aBoard[row]]==1){
          if(left[row+(size-1)-aBoard[row]]==1||right[row+aBoard[row]]==1){
            return;
          }
        }
        int s=symmetryOps(size);
        if(s!=0){ 
          UNIQUE++; 
          TOTAL+=s; 
        }
        row--;
      }
    }else{
      if(aBoard[row]>=0){
        //down[tmp]=left[row+(size-1)-tmp]=right[row+tmp]=0;
        //left[row+(size-1)-tmp]=right[row+tmp]=0;
        left[row+(size-1)-aBoard[row]]=right[row+aBoard[row]]=0;
        aBoard[row]=-1;
      }
      //back tracking
      row--;
    }
  }
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int row,int size){
  bool matched;
  while(row>=0){
    matched=false;
    //search begins at the position previously visited
		int lim=(row!=0)?size:(size+1)/2;
    for(int i=aBoard[row]+1;i<lim;i++){
      //the first matched position
      if(down[i]==0&&left[row+(size-1)-i]==0&&right[row+i]==0){
        //clear original record 
        if(aBoard[row]>=0){
          down[aBoard[row]]=left[row+(size-1)-aBoard[row]]=right[row+aBoard[row]]=0;
        }
        aBoard[row]=i;
        down[i]=left[row+(size-1)-i]=right[row+i]=1;
        matched=true;
        break;
      }
    }
    if(matched){
      //next aBoard
      row++;
      if(row==size){
        if(aBoard[row]!=-1){
          if(down[aBoard[row]]==1||left[row-aBoard[row]+(size-1)]==1||right[row+aBoard[row]]==1){
            return;
          }
        }
        int s=symmetryOps(size);
        if(s!=0){ 
          UNIQUE++; 
          TOTAL+=s; 
        }
        row--;
      }
    }else{
      if(aBoard[row]>=0){
        int tmp=aBoard[row];
        aBoard[row]=-1;
        down[tmp]=left[row+(size-1)-tmp]=right[row+tmp]=0;
      }
      //back tracking
      row--;
    }
  }
}
//CPUR 再帰版 ロジックメソッド
void NQueenR(int row,int size){
  int tmp;
  //枝刈り
  //if(row==size){
  if(row==size-1){
    // 2. 枝刈り
    if(left[row-aBoard[row]+(size-1)]==1||right[row+aBoard[row]]==1){
      return;
    }
    int s=symmetryOps(size);	//対称解除法の導入
    if(s!=0){
      UNIQUE++;
      TOTAL+=s;
    }
  }else{
    // 1. 枝刈り
    //for(int i=0;i<size;i++){
    int lim=(row!=0) ? size : (size+1)/2;
    for(int i=row;i<lim;i++){
      //コメントアウト
      //aBoard[row]=i;
      //交換
      tmp=aBoard[i];
      aBoard[i]=aBoard[row];
      aBoard[row]=tmp;
      if(left[row+(size-1)-aBoard[row]]==0&&right[row+aBoard[row]]==0){
        left[row+(size-1)-aBoard[row]]=right[row+aBoard[row]]=1;
        NQueenR(row+1,size);
        left[row+(size-1)-aBoard[row]]=right[row+aBoard[row]]=0;
      }
    }
	  // 交換
    tmp=aBoard[row];
    for(int i=row;i<lim;i++){
      aBoard[i]=aBoard[i+1];
    }
    aBoard[size-1]=tmp;
	}
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
    printf("\n\n５．CPU 非再帰 バックトラック＋対称解除法＋枝刈りと最適化\n");
  }else if(cpur){
    printf("\n\n５．CPUR 再帰 バックトラック＋対称解除法＋枝刈りと最適化\n");
  }else if(gpu){
    printf("\n\n５．GPU 非再帰 バックトラック\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
		clock_t st;           //速度計測用
		char t[20];           //hh:mm:ss.msを格納
    int min=4; int targetN=18;
    for(int i=min;i<=targetN;i++){
      TOTAL=0; UNIQUE=0;
      st=clock();
      if(cpu){
        //非再帰は-1で初期化
        for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
        NQueen(0,i);
      }
      if(cpur){
        //再帰は0で初期化
        //for(int j=0;j<=targetN;j++){ aBoard[j]=0; } 
        for(int j=0;j<=targetN;j++){ aBoard[j]=j; } 

        NQueenR(0,i);
      }
      TimeFormat(clock()-st,t); 
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
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
