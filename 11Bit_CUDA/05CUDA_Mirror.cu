/**
 *
 * bash版ミラーのC言語版のGPU/CUDA移植版
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
unsigned int down[MAX];   //ポストフラグ/ビットマップ/ミラー
unsigned int left[MAX];   //ポストフラグ/ビットマップ/ミラー
unsigned int right[MAX];  //ポストフラグ/ビットマップ/ミラー
unsigned int bitmap[MAX]; //ミラー
unsigned long COUNT2=0;   //ミラー
//ミラー処理部分 非再帰版
void mirror_solve_NR(unsigned int size,unsigned int row,unsigned int _left,unsigned int _down, unsigned int _right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  left[row]=_left;
  down[row]=_down;
  right[row]=_right;
  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
  while(row>0){
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];
      bitmap[row]=bitmap[row]^bit;
      board[row]=bit;
      if(row==(size-1)){
        COUNT2++;
        row--;
      }else{
        unsigned int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=(down[n]|bit);
        right[row]=(right[n]|bit)>>1;
        board[row]=bit;   //Qを配置
        //クイーンが配置可能な位置を表す
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }
}
// ミラー 非再帰版
void mirror_NR(unsigned int size)
{
  COUNT2=0;
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){ //奇数でも偶数でも通過
    bit=1<<i;
    board[0]=bit;             //１行目にQを置く
    mirror_solve_NR(size,1,bit<<1,bit,bit>>1);
  }
  if(size%2){                 //奇数で通過
    bit=1<<(size-1)/2;
    board[0]=1<<(size-1)/2;   //１行目の中央にQを配置
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      mirror_solve_NR(size,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;    //倍にする
}
//ミラーロジック 再帰版
void mirror_solve_R(unsigned int size,unsigned int row,unsigned int left,unsigned int down,unsigned int right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  if(row==size){
    COUNT2++;
  }else{
    for(unsigned int bitmap=mask&~(left|down|right);bitmap;bitmap=bitmap&~bit){
      bit=-bitmap&bitmap;
      board[row]=bit;
      mirror_solve_R(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
// ミラー 再帰版
void mirror_R(unsigned int size)
{
  COUNT2=0;
  unsigned int bit=0;
  unsigned int limit=size%2 ? size/2-1 : size/2;
  for(unsigned int i=0;i<size/2;++i){
    bit=1<<i;
    board[0]=bit;           //１行目にQを置く
    mirror_solve_R(size,1,bit<<1,bit,bit>>1);
  }
  if(size%2){               //奇数で通過
    bit=1<<(size-1)/2;
    board[0]=1<<(size-1)/2; //１行目の中央にQを配置
    unsigned int left=bit<<1;
    unsigned int down=bit;
    unsigned int right=bit>>1;
    for(unsigned int i=0;i<limit;++i){
      bit=1<<i;
      mirror_solve_R(size,2,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  TOTAL=COUNT2<<1;    //倍にする
}
// ビットマップ 非再帰版
void bitmap_NR(unsigned int size,int row)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bitmap[size];
  unsigned int bit=0;
  bitmap[row]=mask;
  while(row>-1){
    if(bitmap[row]>0){
      bit=-bitmap[row]&bitmap[row];//一番右のビットを取り出す
      bitmap[row]=bitmap[row]^bit;//配置可能なパターンが一つずつ取り出される
      board[row]=bit;
      if(row==(size-1)){
        TOTAL++;
        row--;
      }else{
        unsigned int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=down[n]|bit;
        right[row]=(right[n]|bit)>>1;
        board[row]=bit;
        //クイーンが配置可能な位置を表す
        bitmap[row]=mask&~(left[row]|down[row]|right[row]);
      }
    }else{
      row--;
    }
  }//end while
}
// ビットマップ 再帰版
void bitmap_R(unsigned int size,unsigned int row,unsigned int left,unsigned int down, unsigned int right)
{
  unsigned int mask=(1<<size)-1;
  unsigned int bit=0;
  if(row==size){
    TOTAL++;
  }else{
    // クイーンが配置可能な位置を表す
    for(unsigned int bitmap=mask&~(left|down|right);bitmap;bitmap=bitmap&~bit){
      bit=bitmap&-bitmap;
      board[row]=bit;
      bitmap_R(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
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
// バックトラック　効き筋をチェック
int check_backTracking(unsigned int size,unsigned int row)
{
  for(unsigned int i=0;i<row;++i){
    unsigned int val=0;
    if(board[i]>=board[row]){
      val=board[i]-board[row];
    }else{
      val=board[row]-board[i];
    }
    if(board[i]==board[row]||val==(row-i)){
      return 0;
    }
  }
  return 1;
}
// バックトラック 非再帰版
void backTracking_NR(unsigned int size,int row)
{
  // １．非再帰は初期化が必要
  for(unsigned int i=0;i<size;++i){
    board[i]=-1;
  }
  // ２．再帰で呼び出される関数内を回す処理
  while(row>-1){
    unsigned int matched=0;   //クイーンを配置したか
    // ３．再帰処理のループ部分
    for(unsigned int col=board[row]+1;col<size;++col){
      board[row]=col;   // クイーンを配置
      // 効きをチェック
      if(check_backTracking(size,row)==1){
        matched=1;
        break;
      } // end if
    } // end for
    // ４．配置したら実行したい処理
    if(matched){
      row++;
      // ５．最下部まで到達したときの処理
      if(row==size){
        row--;
        TOTAL++;
      }
    // ６．配置できなくてバックトラックしたい時の処理
    }else{
      if(board[row]!=-1){
        board[row]=-1;
      }
      row--;
    }
  } //end while
}
// バックトラック 再帰版
void backTracking_R(unsigned int size,int row)
{
  if(row==size){
    TOTAL++;
  }else{
    for(unsigned int col=0;col<size;++col){
      board[row]=col;
      if(check_backTracking(size,row)==1){
        backTracking_R(size,row+1);
      }
    }// end for
  }//end if
}
// ブルートフォース 効き筋をチェック
int check_bluteForce(unsigned int size)
{
  for(unsigned int r=1;r<size;++r){
    unsigned int val=0;
    for(unsigned int i=0;i<r;++i){
      if(board[i]>=board[r]){
        val=board[i]-board[r];
      }else{
        val=board[r]-board[i];
      }
      if(board[i]==board[r]||val==(r-i)){
        return 0;
      }
    }
  }
  return 1;
}
//ブルートフォース 非再帰版
void bluteForce_NR(unsigned int size,int row)
{
  // １．非再帰は初期化が必要
  for(unsigned int i=0;i<size;++i){
    board[i]=-1;
  }
  // ２．再帰で呼び出される関数内を回す処理
  while(row>-1){
    unsigned int matched=0;   //クイーンを配置したか
    // ３．再帰処理のループ部分
    // 非再帰では過去の譜石を記憶するためにboard配列を使う
    for(unsigned int col=board[row]+1;col<size;++col){
      board[row]=col;
      matched=1;
      break;
    }
    // ４．配置したら実行したい処理
    if(matched){
      row++;
      // ５．最下部まで到達したときの処理';
      if(row==size){
        row--;
        // 効きをチェック
        if(check_bluteForce(size)==1){
          TOTAL++;
        }
      }
      // ６．配置できなくてバックトラックしたい処理
    }else{
      if(board[row]!=-1){
        board[row]=-1;
      }
      row--;
    } // end if
  }//end while
}
//ブルートフォース 再帰版
void bluteForce_R(unsigned int size,int row)
{
  if(row==size){
    if(check_bluteForce(size)==1){
      TOTAL++; // グローバル変数
    }
  }else{
    for(int col=0;col<size;++col){
      board[row]=col;
      bluteForce_R(size,row+1);
    }
  }
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
  if(cpu){ printf("\n\nミラー \n"); }
  else if(cpur){ printf("\n\nミラー \n"); }
  else if(gpu){ printf("\n\nミラー \n"); }
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
        // postFlag_R(size,0);     //配置フラグ
        // bitmap_R(size,0,0,0,0); //ビットマップ
        mirror_R(size);         //ミラー
      }
      if(cpu){ //非再帰
        //bluteForce_NR(size,0);//ブルートフォース
        // backTracking_NR(size,0);//バックトラック
        // postFlag_NR(size,0);     //配置フラグ
        // bitmap_NR(size,0);  //ビットマップ
        mirror_NR(size);         //ミラー
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
