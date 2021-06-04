/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイルと実行
 $ nvcc CUDA**_N-Queen.cu && ./a.out (-c|-r|-g|-s)
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
#define THREAD_NUM		96
//#define THREAD_NUM		1
#define MAX 27
unsigned int NONE=3;
unsigned int POINT=2;
unsigned int ROTATE=1;
long cnt[3];
long pre[3];
//変数宣言
long TOTAL=0; //GPU,CPUで使用
/***07 uniq*************************************/
long UNIQUE=0;//GPU,CPUで使用
/****************************************/
typedef unsigned long long uint64;

//ローカル構造体
typedef struct{
  uint64 bv;
  uint64 bh;
  uint64 bu;
  uint64 bd;
  int x[MAX];
  int y[MAX];
}Board ;
Board B;

//CPU/GPU
//hh:mm:ss.ms形式に処理時間を出力
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
void process(int si,Board B,int sym){
    printf("process\n");

}
//
bool board_placement(int si,int x,int y,int pos){
   //同じ場所に置くかチェック
   for(int i=0;i<8;i++){
       //printf("i:%d:x:%d:y:%d\n",i,B.x[i],B.y[i]);
       if(B.x[i]==x&&B.y[i]==y){
         printf("Duplicate x:%d:y:%d\n",x,y);
         return true;  
       }
   }
   B.x[pos]=x;
   B.y[pos]=y;
   uint64 bv=1<<x;
   uint64 bh=1<<y;
   uint64 bu=1<<(si-1-x+y);
   uint64 bd=1<<(x+y);
   printf("check valid x:%d:y:%d:p.N-1-x+y:%d;x+y:%d\n",x,y,si-1-x+y,x+y);
   printf("check valid pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
   printf("bvcheck:%d:bhcheck:%d:bucheck:%d:bdcheck:%d\n",B.bv&bv,B.bh&bh,B.bu&bu,B.bd&bd);
   if((B.bv&bv)||(B.bh&bh)||(B.bu&bu)||(B.bd&bd)){
	   printf("valid_false\n");
       return false;
   }     
   printf("before pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
   B.bv |=bv;
   B.bh |=bh;
   B.bu |=bu;
   B.bd |=bd;
   printf("after pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
   printf("valid_true\n");
   return true;
}
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size)
{
  
  int pres_a[930];
  int pres_b[930];
  int idx=0;
  for(int a=0;a<size;a++){
      for(int b=0;b<size;b++){
        if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){
          continue;
        }     
        pres_a[idx]=a;
        pres_b[idx]=b;
        idx++;
      }
  }
  printf("idx:%d\n",idx);
  printf("(N/2)*(N-3):%d\n",(size/2)*(size-3));
  for(int w=0;w<=(size/2)*(size-3);w++){
      B.bv=0;
      B.bh=0;
      B.bu=0;
      B.bd=0;
      for(int i=0;i<8;i++){
          B.x[i]=size;
          B.y[i]=size;
      }
      int wa=pres_a[w];
      int wb=pres_b[w];
      printf("wloop:w:%d:p.a:%d,p.b:%d:wa:%d:wb:%d\n",w,pres_a[w],pres_b[w],wa,wb);
      printf("placement_pwa:xk(0):0:y:%d\n",wa);
      board_placement(size,0,wa,0);
      printf("placement_pwb:xk(1):1:y:%d\n",wb);
      board_placement(size,1,wb,1);
      Board nB=B;
      for(int  n = w; n < (size-2)*(size-1)-w; n++) {
         printf("nloop:n:%d\n",n);
         int na=pres_a[n];
         int nb=pres_b[n];   
         printf("placement_pna:x:%d:yk(N-1):%d\n",na,size-1);
         bool pna=board_placement(size,na,size-1,2);
         if(pna==false){
             printf("pnaskip:na:%d:N-1:%d\n",na,size-1);
             B=nB;
             continue;
         }
         printf("placement_pnb:x:%d:yk(N-2):%d\n",nb,size-2);
         bool pnb=board_placement(size,nb,size-2,3);
         if(pnb==false){
             printf("pnbskip:nb:%d:N-2:%d\n",nb,size-2);
             B=nB;
             continue;
         }
         Board eB=B;
         for(int  e = w; e < (size-2)*(size-1)-w; e++) {
            printf("eloop:e:%d\n",e);
            int ea=pres_a[e];
            int eb=pres_b[e];
            printf("placement_pea:xk(N-1):%d:y:%d\n",size-1,size-1-ea);
            bool pea=board_placement(size,size-1,size-1-ea,4);
            if(pea==false){
              B=eB;
              printf("peaskip:N-1:%d:N-1-ea:%d\n",size-1,size-1-ea);
              continue;
            }
            printf("placement_peb:xk(N-2):%d:y:%d\n",size-2,size-1-eb);
            bool peb=board_placement(size,size-2,size-1-eb,5);
            if(peb==false){
              printf("pebskip:N-2:%d:N-1-eb:%d\n",size-2,size-1-eb);
              B=eB;
              continue;
            }
            Board sB=B;
            for(int s = w; s < (size-2)*(size-1)-w; s++) {
               printf("sloop:s:%d\n",s);
               int sa =pres_a[s];
               int sb =pres_b[s];
               printf("psa:x:%d:yk(0):0\n",size-1-sa);
               bool psa=board_placement(size,size-1-sa,0,6);
               if(psa==false){
                B=sB;
                printf("psaskip:N-1-sa:%d:0\n",size-1-sa);
                continue;
               }
               printf("psb:x:%d:yk(1):1\n",size-1-sb);
               bool psb=board_placement(size,size-1-sb,1,7);
               if(psb==false){
                B=sB;
                printf("psbskip:N-1-sb:%d:1\n",size-1-sb);
                continue;
               }
              printf("noskip\n");
              printf("pwa:xk(0):0:y:%d\n",wa);
              printf("pwb:xk(1):1:y:%d\n",wb);
              printf("pna:x:%d:yk(N-1):%d\n",na,size-1);
              printf("pnb:x:%d:yk(N-2):%d\n",nb,size-2);
              printf("pea:xk(N-1):%d:y:%d\n",size-1,size-1-ea);
              printf("peb:xk(N-2):%d:y:%d\n",size-2,size-1-eb);
              printf("psa:x:%d:yk(0):0\n",size-1-sa);
              printf("psb:x:%d:yk(1):1\n",size-1-sb);
               //
               int ww = (size-2)*(size-1)-1-w;
               if(s == ww) {
                if(n < (size-2)*(size-1)-1-e) {
                  B=sB;
                  continue;
                }
               }
               if(e == ww) {
                if(n > (size-2)*(size-1)-1-n) {
                  B=sB;
                  continue;       
                }
               }
               if(n == ww) {
                 if(e > (size-2)*(size-1)-1-s) {
                   B=sB;
                   continue;
                 }
               }
               //
               if(s==w){
                   if((n != w) || (e != w)) {
                    B=sB;
                    continue;
                   }
                   process(size,B,ROTATE);
                   //(*act)(board, Symmetry::ROTATE);

               }
               if(e == w) {
                if(n >= s) {
                  if(n > s) {
                    B=sB;
                    continue;
                  }
                }
                process(size,B,POINT);
                //(*act)(board, Symmetry::POINT);   
               }
               process(size,B,NONE);
               //(*act)(board, Symmetry::NONE);
               B=sB;
            }
         }    
      }
  }
  int bit=0;
  /***07 aBoardローカル化*********************/
  unsigned int aBoard[MAX];
  /************************/  
  //1行目全てにクイーンを置く
  for(int col=0;col<size;col++){
    aBoard[0]=bit=(1<<col);
    /***07 aBoardローカル化*********************/
    //solve_nqueenr(size,mask,1,bit<<1,bit,bit>>1,aBoard);
    /************************/  
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
    int min=8;int targetN=8;
    int mask;
    for(int i=min;i<=targetN;i++){
      /***07 symmetryOps CPU,GPU同一化*********************/
      TOTAL=0; UNIQUE=0;
      //COUNT2=COUNT4=COUNT8=0;
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
    int min=5;int targetN=17;
    //int min=8;int targetN=8;
   
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

