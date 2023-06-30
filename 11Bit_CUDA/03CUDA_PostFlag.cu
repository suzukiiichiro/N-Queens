/**
 *
 * bash版配置フラグのC言語版のGPU/CUDA移植版
 *
 詳しい説明はこちらをどうぞ
 https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題
 *
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define THREAD_NUM		96
#define MAX 27
// システムによって以下のマクロが必要であればコメントを外してください。
//#define UINT64_C(c) c ## ULL
//
// グローバル変数
unsigned long TOTAL=0; 
unsigned long UNIQUE=0;
int board[MAX];  //ボード配列
unsigned int down[MAX];   //ポストフラグ
unsigned int left[MAX];   //ポストフラグ
unsigned int right[MAX];  //ポストフラグ
// ポストフラグ 非再帰版
void postFlag_NR(unsigned int size,int row)
{
  // １．非再帰は初期化が必要
  for(unsigned int i=0;i<size;++i){
    board[i]=-1;
  }
  // ２．再帰で呼び出される関数内を回す処理
  while(row>-1){
    unsigned int matched=0; //クイーンを配置したか
    // ３．再帰処理のループ部分
    // 非再帰では過去の譜石を記憶するためにboard配列を使う
    for(unsigned int col=board[row]+1;col<size;++col){
      if(!down[col]
          && !right[col-row+size-1]
          && !left[col+row]){
        // unsigned int dix=col;
        // unsigned int rix=row-col+(size-1);
        // unsigned int lix=row+col;
        /** バックトラックではここで効きをチェックしていた
        check_backTracking "$row";  # 効きをチェック
        */
        // 効きとしてフラグをfalseにする
        if(board[row]!=-1){
          down[board[row]]=0;
          right[board[row]-row+(size-1)]=0;
          left[board[row]+row]=0;
        }
        board[row]=col; //クイーンを配置
        // 効きを開放（trueに）する
        down[col]=1;
        right[col-row+(size-1)]=1;
        left[col+row]=1;
        matched=1;
        break;
      } //end if
    }//end for
    // ４．配置したら実行したい処理
    if(matched){
      row++;
      // ５．最下部まで到達したときの処理
      if(row==size){
        row--;
        /** ブルートフォースではここで効きをチェックしていた
        // check_bluteForce "$size";   # 効きをチェック
        */
        TOTAL++;
      }
    // ６．配置できなくてバックトラックしたい時の処理
    }else{
      if(board[row]!=-1){
        down[board[row]]=0;
        right[board[row]-row+(size-1)]=0;
        left[board[row]+row]=0;
        board[row]=-1;
      }
      row--;
    }
  }//end while
}
// ポストフラグ 再帰版
void postFlag_R(unsigned int size,int row)
{
  if(row==size){
    TOTAL++;
  }else{
    for(unsigned int col=0;col<size;++col){
      board[row]=col;
      if(down[col]==0
          && right[row-col+size-1]==0
          && left[row+col]==0){
        down[col]=1;
        right[row-col+(size-1)]=1;
        left[row+col]=1;
        postFlag_R(size,row+1);
        down[col]=0;
        right[row-col+(size-1)]=0;
        left[row+col]=0;
      }//end if
    }//end for
  }//end if
}
// CUDA 初期化
bool InitCUDA()
{
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  int i;
  for(i=0;i<count;i++){
    struct cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
//メイン
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else{ gpu=true; } //デフォルトをgpuとする
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s] n steps\n",argv[0]);
    printf("  -r: CPUR only\n");
    printf("  -c: CPU only\n");
    printf("  -g: GPU only\n");
  }
  if(cpu){ printf("\n\n配置フラグ 非再帰 \n"); }
  else if(cpur){ printf("\n\n配置フラグ 再帰 \n"); }
  else if(gpu){ printf("\n\n配置フラグ GPGPU/CUDA \n"); }
  if(cpu||cpur){
    int min=4; 
    int targetN=17;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      TOTAL=UNIQUE=0;
      gettimeofday(&t0, NULL);//計測開始
      if(cpur){ //再帰
        // bluteForce_R(size,0);//ブルートフォース
        // backTracking_R(size,0); //バックトラック
        postFlag_R(size,0);     //配置フラグ
      }
      if(cpu){ //非再帰
        //bluteForce_NR(size,0);//ブルートフォース
        // backTracking_NR(size,0);//バックトラック
        postFlag_NR(size,0);     //配置フラグ
      }
      //
      gettimeofday(&t1, NULL);//計測終了
      int ss;int ms;int dd;
      if(t1.tv_usec<t0.tv_usec) {
        dd=(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else {
        dd=(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n",
          size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    } //end for
  }//end if
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    /* int steps=24576; */
    int min=4;
    int targetN=21;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        TOTAL=UNIQUE=0;
        //
      }
      gettimeofday(&t1,NULL);   // 計測終了
      int ss;int ms;int dd;
      if (t1.tv_usec<t0.tv_usec) {
        dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      } else {
        dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n",
          i,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }//end for
  }//end if
  return 0;
}
