/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 nq27_N-Queen.c && ./a.out [-c|-r]
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 
                    -s SGPU(サマーズ版と思われる)
                    
**/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define THREAD_NUM		96
//#define THREAD_NUM		1
#define MAX 27
//
typedef unsigned long long uint64;
typedef struct{
  uint64 bv;
  uint64 down;
  uint64 left;
  uint64 right;
  int x[MAX];
}Board ;
//
Board B;
unsigned int COUNT8=2;
unsigned int COUNT4=1;
unsigned int COUNT2=0;
long cnt[3];
long pre[3];
//変数宣言
long TOTAL=0; //GPU,CPUで使用
long UNIQUE=0;//GPU,CPUで使用
//
void TimeFormat(clock_t utime,char *form)
{
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
//出力
void dec2bin(int size, int dec)
{
  int i, b[32];
  for (i = 0; i < size; i++)
  {
    b[i] = dec % 2;
    dec = dec / 2;
  }
  while (i > 0)
    printf("%1d", b[--i]);
}
void breakpoint(int size,char* string,int* x,int row)
{
  printf("%s\n",string);
  printf("<>N=%d STEP:",size);
  for(int i=0;i<size;i++){
    if(x[i]==-1){ 
      printf("-"); 
    }else{
      printf("%d",x[i]);
    }
  }
  printf("  ");
  printf("row:%d\n",row);
  printf("\n");
  //colの座標表示
  printf("   ");
  for (int j=size-1;j>=0;j--){
    printf(" %2d",j);
  }
  printf("\n");
  printf(" =============");
  for (int j=0;j<size;j++){
    printf("==");
  }
  printf("\n");
  //row
  for (int j=0;j<size;j++){
    printf("%2d| ",j);
    if(x[j]==-1){ dec2bin(size,0); }
    else{ 
      dec2bin(size,1<<x[j]); 
    }
    printf("\n");
  }
  printf("\n");
  /**
   *
   *
   */
  //int moji;
	//while ((moji = getchar()) != EOF){
//		switch (moji){
//		case '\n':
//		  return;
//		default:
//			break;
//		}
//	}
}


void NQueenR(int size)
{
  int pres_a[930];
  int pres_b[930];
  long mask[930];
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){
        continue;
      }     
      pres_a[idx]=a;
      mask[idx]=mask|1<<(size-1-0+a)|1<<a|1<<(0+a);
      pres_b[idx]=b;
      mask[idx]=mask|1<<(size-1-1+b)|1<<b|1<<(1+b);
      idx++;
    }
  }
  printf("idx:%d\n",idx);
  for(int i=0;i<idx;i++){
    for(int j=0;j<idx;j++){
    }
  }
}
//メインメソッド
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1,steps=24576;
  //int argstart=1,steps=1;

  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else
      cpur=true;
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){
    printf("\n\n７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(cpur){
    printf("\n\n７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(gpu){
    printf("\n\n７．GPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }else if(sgpu){
    printf("\n\n７．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;           //速度計測用
    char t[20];           //hh:mm:ss.msを格納
    //int min=5; int targetN=17;
    int min=4;int targetN=17;
    min=8;
    targetN=8;
    int mask;
    for(int i=min;i<=targetN;i++){
      /***07 symmetryOps CPU,GPU同一化*********************/
      TOTAL=0; UNIQUE=0;
      //COUNT2=COUNT4=COUNT8=0;
      for(int j=0;j<=2;j++){
        pre[j]=0;
        cnt[j]=0;
      }
      /************************/
      mask=(1<<i)-1;
      st=clock();
      //
      //【通常版】
      //if(cpur){ _NQueenR(i,mask,0,0,0,0); }
      //CPUR
      if(cpur){ 
        NQueenR(i); 
        //printf("通常版\n");
      }
      //CPU
      if(cpu){ 
        //NQueen(i,mask); 
        printf("準備中\n");
      }
      //
      TimeFormat(clock()-st,t); 
      /***07 symmetryOps CPU,GPU同一化*********************/
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
      /************************/
    }
  }
  if(gpu||sgpu){
    int min=4;int targetN=17;
    struct timeval t0;struct timeval t1;
    int ss;int ms;int dd;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);   // 計測開始
      if(gpu){
        TOTAL=0;
        UNIQUE=0;
        //NQueenG(i,steps);
      }else if(sgpu){
        printf("準備中");
        //TOTAL=sgpu_solve_nqueen_cuda(i,steps);
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
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n", i,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}

