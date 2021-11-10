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
  uint64 bh;
  uint64 bu;
  uint64 bd;
  int x[MAX];
}Board ;
//
Board B;
unsigned int NONE=2;
unsigned int POINT=1;
unsigned int ROTATE=0;
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
//
long countCompletions(uint64_t bv,uint64_t bh,uint64_t bu,uint64_t  bd)
{
  // Placement Complete?
  //printf("countCompletions_start\n");
  //printf("bv:%d\n",bv);
  //printf("bh:%d\n",bh);
  //printf("bu:%d\n",bu);
  //printf("bd:%d\n",bd);
  //bh=-1 1111111111 すべての列にクイーンを置けると-1になる
  if(bh+1==0){
    //printf("return_bh+1==0:%d\n",bh);  
    return  1;
  }
  // -> at least one more queen to place
  while((bv&1)!=0) { // Column is covered by pre-placement
    //bv 右端にクイーンがすでに置かれていたら。クイーンを置かずに１行下に移動する
    //bvを右端から１ビットずつ削っていく。ここではbvはすでにクイーンが置かれているかどうかだけで使う
    bv>>=1;//右に１ビットシフト
    bu<<=1;//left 左に１ビットシフト
    bd>>=1;//right 右に１ビットシフト
    //printf("while:bv:%d\n",bv);
    //printf("while:bu:%d\n",bu);
    //printf("while:bd:%d\n",bd);
    //printf("while:bv&1:%d\n",bv&1);
  }
  //１行下に移動する
  bv>>=1;
  //printf("onemore_bv:%d\n",bv);
  //printf("onemore_bh:%d\n",bh);
  //printf("onemore_bu:%d\n",bu);
  //printf("onemore_bd:%d\n",bd);
  //
  // Column needs to be placed
  long  cnt=0;
  uint64_t slot;
  //bh:down bu:left bd:right
  //クイーンを置いていく
  //slotsはクイーンの置ける場所
  for(uint64_t slots=~(bh|bu|bd);slots!=0;) {
    //printf("colunm needs to be placed\n");
    //printf("slots:%d\n",slots);
    slot=slots&-slots;
    //printf("slot:%d\n",slot);
    //printf("bv:%d:bh|slot:%d:(bu|slot)<<1:%d:(bd|slot)>>1:%d\n",bv,bh|slot,(bu|slot)<<1,(bd|slot)>>1);
    cnt+=countCompletions(bv,bh|slot,(bu|slot)<<1,(bd|slot)>>1);
    slots^=slot;
    //printf("slots:%d\n",slots);
  }
  //途中でクイーンを置くところがなくなるとここに来る
  //printf("return_cnt:%d\n",cnt);
  return cnt;
} // countCompletions()
//
void process(int si,Board B,int sym)
{
  //printf("process\n");
  pre[sym]++;
  //printf("N:%d\n",si);
  //BVは行 x 
  //printf("getBV:%d\n",B.bv);
  //BHはdown y
  //printf("getBH:%d\n",B.bh);
  //BU left N-1-x+y 右上から左下
  //printf("getBU:%d\n",B.bu);
  //BD right x+y 左上から右下
  //printf("getBD:%d\n",B.bd);
  //printf("before_cnt_sym:%d\n",cnt[sym]);
  cnt[sym] += countCompletions(B.bv >> 2,
      ((((B.bh>>2)|(~0<<(si-4)))+1)<<(si-5))-1,
      B.bu>>4,
      (B.bd>>4)<<(si-5));

  //行 brd.getBV()>>2 右2ビット削除 すでに上２行はクイーンを置いているので進める BVは右端を１ビットずつ削っていく
  //列 down ((((brd.getBH()>>2)|(~0<<(N-4)))+1)<<(brd.N-5))-1 8だと左に1シフト 9:2 10:3 
  //brd.getBU()>>4 left  右４ビット削除
  //(brd.getBD()>>4)<<(N-5)) right 右４ビット削除後N-5個分左にシフト
  //printf("cnt_sym:%d\n",cnt[sym]);
}
//
bool board_placement(int si,int x,int y)
{
  //同じ場所に置くかチェック
  //printf("i:%d:x:%d:y:%d\n",i,B.x[i],B.y[i]);
  if(B.x[x]==y){
    //printf("Duplicate x:%d:y:%d\n",x,y);
    return true;  
  }
  B.x[x]=y;
  uint64 bv=1<<x;
  uint64 bh=1<<y;
  uint64 bu=1<<(si-1-x+y);
  uint64 bd=1<<(x+y);
  //printf("check valid x:%d:y:%d:p.N-1-x+y:%d;x+y:%d\n",x,y,si-1-x+y,x+y);
  //printf("check valid pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("bvcheck:%d:bhcheck:%d:bucheck:%d:bdcheck:%d\n",B.bv&bv,B.bh&bh,B.bu&bu,B.bd&bd);
  if((B.bv&bv)||(B.bh&bh)||(B.bu&bu)||(B.bd&bd)){
    //printf("valid_false\n");
    return false;
  }     
  //printf("before pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  B.bv |=bv;
  B.bh |=bh;
  B.bu |=bu;
  B.bd |=bd;
  //printf("after pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("valid_true\n");
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
  //printf("idx:%d\n",idx);
  //printf("(N/2)*(N-3):%d\n",(size/2)*(size-3));

  //プログレス
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);

  Board wB=B;
  for(int w=0;w<=(size/2)*(size-3);w++){
    B=wB;
    //
    // B.bv=0;
    // B.bh=0;
    // B.bu=0;
    // B.bd=0;
    B.bv=B.bh=B.bu=B.bd=0;
    //
    for(int i=0;i<size;i++){
      B.x[i]=-1;
    }
    // 不要
    // int wa=pres_a[w];
    // int wb=pres_b[w];
    //
    //printf("wloop:w:%d:p.a:%d,p.b:%d:wa:%d:wb:%d\n",w,pres_a[w],pres_b[w],wa,wb);
    //printf("placement_pwa:xk(0):0:y:%d\n",wa);

    //プログレス
    printf("\r(%d/%d)",w,((size/2)*(size-3)));// << std::flush;
    printf("\r");
    fflush(stdout);
  
    //
    // 謎１
    // 
    // 置き換え
    // board_placement(size,0,wa);
    board_placement(size,0,pres_a[w]);
    //printf("placement_pwb:xk(1):1:y:%d\n",wb);
    //
    //
    // 謎２
    // 
    //置き換え
    //board_placement(size,1,wb);
    board_placement(size,1,pres_b[w]);

    Board nB=B;
    //追加
    int lsize=(size-2)*(size-1)-w;
    //for(int n=w;n<(size-2)*(size-1)-w;n++){
    for(int n=w;n<lsize;n++){
      B=nB;
      //printf("nloop:n:%d\n",n);
      //
      // 不要
      // int na=pres_a[n];
      // int nb=pres_b[n];   
      //
      //printf("placement_pwa:xk(0):0:y:%d\n",wa);
      //printf("placement_pwb:xk(1):1:y:%d\n",wb);
      //printf("placement_pna:x:%d:yk(N-1):%d\n",na,size-1);
      //置き換え
      //bool pna=board_placement(size,na,size-1);
      //bool pna=board_placement(size,pres_a[n],size-1);
      //インライン
      //if(pna==false){
      if(board_placement(size,pres_a[n],size-1)==false){
        //printf("pnaskip:na:%d:N-1:%d\n",na,size-1);
        continue;
      }
      //printf("placement_pnb:x:%d:yk(N-2):%d\n",nb,size-2);
      //置き換え
      //bool pnb=board_placement(size,nb,size-2);
      //bool pnb=board_placement(size,pres_b[n],size-2);
      //インライン
      //if(pnb==false){
      if(board_placement(size,pres_b[n],size-2)==false){
        //printf("pnbskip:nb:%d:N-2:%d\n",nb,size-2);
        continue;
      }
      Board eB=B;
      //for(int e=w;e<(size-2)*(size-1)-w;e++){
      for(int e=w;e<lsize;e++){
        B=eB;
        //printf("eloop:e:%d\n",e);
        //不要
        //int ea=pres_a[e];
        //int eb=pres_b[e];
        //printf("placement_pwa:xk(0):0:y:%d\n",wa);
        //printf("placement_pwb:xk(1):1:y:%d\n",wb);
        //printf("placement_pna:x:%d:yk(N-1):%d\n",na,size-1);
        //printf("placement_pnb:x:%d:yk(N-2):%d\n",nb,size-2);
        //printf("placement_pea:xk(N-1):%d:y:%d\n",size-1,size-1-ea);
        //置き換え
        //bool pea=board_placement(size,size-1,size-1-ea);
        //インライン
        //if(pea==false){
        if(board_placement(size,size-1,size-1-pres_a[e])==false){
          //printf("peaskip:N-1:%d:N-1-ea:%d\n",size-1,size-1-ea);
          continue;
        }
        //printf("placement_peb:xk(N-2):%d:y:%d\n",size-2,size-1-eb);
        //置き換え
        //bool peb=board_placement(size,size-2,size-1-eb);
        //インライン
        //if(peb==false){
        if(board_placement(size,size-2,size-1-pres_b[e])==false){
          //printf("pebskip:N-2:%d:N-1-eb:%d\n",size-2,size-1-eb);
          continue;
        }
        Board sB=B;
        //for(int s=w;s<(size-2)*(size-1)-w;s++){
        for(int s=w;s<lsize;s++){
          B=sB;
          //printf("sloop:s:%d\n",s);
          //
          //不要
          //int sa =pres_a[s];
          //int sb =pres_b[s];
          //
          //printf("placement_pwa:xk(0):0:y:%d\n",wa);
          //printf("placement_pwb:xk(1):1:y:%d\n",wb);
          //printf("placement_pna:x:%d:yk(N-1):%d\n",na,size-1);
          //printf("placement_pnb:x:%d:yk(N-2):%d\n",nb,size-2);
          //printf("placement_pea:xk(N-1):%d:y:%d\n",size-1,size-1-ea);
          //printf("placement_peb:xk(N-2):%d:y:%d\n",size-2,size-1-eb);
          //printf("psa:x:%d:yk(0):0\n",size-1-sa);
          //置き換え
          //bool psa=board_placement(size,size-1-sa,0);
          //インライン
          //if(psa==false){
          if(board_placement(size,size-1-pres_a[s],0)==false){
            //printf("psaskip:N-1-sa:%d:0\n",size-1-sa);
            continue;
          }
          //printf("psb:x:%d:yk(1):1\n",size-1-sb);
          //bool psb=board_placement(size,size-1-sb,1);
          //if(psb==false){
          if(board_placement(size,size-1-pres_b[s],1)==false){
            //printf("psbskip:N-1-sb:%d:1\n",size-1-sb);
            continue;
          }
          //printf("noskip\n");
          //printf("pwa:xk(0):0:y:%d\n",wa);
          //printf("pwb:xk(1):1:y:%d\n",wb);
          //printf("pna:x:%d:yk(N-1):%d\n",na,size-1);
          //printf("pnb:x:%d:yk(N-2):%d\n",nb,size-2);
          //printf("pea:xk(N-1):%d:y:%d\n",size-1,size-1-ea);
          //printf("peb:xk(N-2):%d:y:%d\n",size-2,size-1-eb);
          //printf("psa:x:%d:yk(0):0\n",size-1-sa);
          //printf("psb:x:%d:yk(1):1\n",size-1-sb);
          //
          int ww=(size-2)*(size-1)-1-w;
          //新設
          int w2=(size-2)*(size-1)-1;
          //if(s==ww){
          if((s==ww)&&(n<(w2-e))){
            //if(n<(size-2)*(size-1)-1-e){
            //if(n<(w2-e)){
              continue;
            //}
          }
          //if(e==ww){
          if((e==ww)&&(n>(w2-n))){
            //if(n>(size-2)*(size-1)-1-n){
            //if(n>(w2-n)){
              continue;       
            //}
          }
          //if(n==ww){
          if((n==ww)&&(e>(w2-s))){
            //if(e>(size-2)*(size-1)-1-s){
            //if(e>(w2-s)){
              continue;
            //}
          }
          if(s==w){
            if((n!=w)||(e!=w)){
              continue;
            }
            process(size,B,ROTATE);
            //(*act)(board, Symmetry::ROTATE);
            continue;
          }
          if((e==w)&&(n>=s)){
            //if(n>=s){
              if(n>s){
                continue;
              }
              process(size,B,POINT);
              //(*act)(board, Symmetry::POINT);   
              continue;
            //}
          }
          process(size,B,NONE);
          //(*act)(board, Symmetry::NONE);
          continue;
        }
      }    
    }
  }
  //printf("ROTATE_0:%d\n",cnt[ROTATE]);
  //printf("POINT_1:%d\n",cnt[POINT]);
  //printf("NONE_2:%d\n",cnt[NONE]);
  UNIQUE=cnt[ROTATE]+cnt[POINT]+cnt[NONE];
  TOTAL=cnt[ROTATE]*2+cnt[POINT]*4+cnt[NONE]*8;
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
