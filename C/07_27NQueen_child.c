/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
<>27. デバッグ

*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MAX 17 
#define DEBUG 0

int si;  
int siE;
int B1;
int B2;
long lTotal;
long lUnique;

/** スレッドローカル構造体 */
typedef struct{
	int bit;
  int own;
	int ptn;
	int you;
  int B1;
  int B2;
  int TB;
  int EB;
  int msk;
  int SM;
  int LM;
  int aB[MAX]; 
  long C2[MAX];
  long C4[MAX];
  long C8[MAX];
}local ;

void symmetryOps_bm(local *l);
void backTrack2(int y,int left,int down,int right,int bm,local *l);
void backTrack1(int y,int left,int down,int right,int bm,local *l);
void NQueen();


#ifdef DEBUG
const int spc[]={'/', '-', '\\', '|'};
const int spl=sizeof(spc)/sizeof(spc[0]);
void thMonitor(local *l,int i);
void hoge();
void hoge(){
  clock_t t;
  t = clock() + CLOCKS_PER_SEC/10;
  while(t>clock());
}
#endif

void thMonitor(local *l,int i){
  printf("\033[G");
  if(i==2){
    printf("\rN:%2d C2[%c] C4[ ] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C2[l->B1]%spl],l->B1,l->B2);
  }
  else if(i==4){
    printf("\rN:%2d C2[ ] C4[%c] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C4[l->B1]%spl],l->B1,l->B2);
  }
  else if(i==8){
    printf("\rN:%2d C2[ ] C4[ ] C8[%c] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1]%spl],l->B1,l->B2);
  }
  else if(i==82){ 
    printf("\rN:%2d C2[ ] C4[ ] C8[ ] C8BT[%c] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1]%spl],l->B1,l->B2);
  }
  printf("\033[G");
  printf("\n");
  for (int y=0;y<si;y++) {
    for (l->bit=l->TB; l->bit; l->bit>>=1){
        char c;
        if(l->aB[y]==l->bit){
          c='Q';
        }else{
          c='-';
        }
        putchar(c);
    }
    printf("|\n");
  }
  printf("\n\n");
}

void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
	l->C8[l->B1]++;
  if(DEBUG>0) thMonitor(l,8); 
  //90度回転
  if(l->aB[l->B2]==1){ 
    //l->own=1; l->ptn=2;
    //while(l->own<=siE){ 
    for(l->own=1,l->ptn=2;l->own<=siE;l->own++,l->ptn<<=1){ 
      //l->bit=1; l->you=siE;
      //while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){
      for(l->bit=1,l->you=siE;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you--);
      //{
      //   l->bit<<=1; l->you--; 
      //}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1]--; return; } 
      //if(l->aB[l->own]<l->bit){ break; }
      else if(l->aB[l->own]<l->bit){ break; }
      //l->own++; l->ptn<<=1; 
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(l->own>siE){ 
			l->C2[l->B1]++;
      l->C8[l->B1]--;
      if(DEBUG>0) thMonitor(l,2);
      return ; 
    } 
  }
  //180度回転
  if(l->aB[siE]==l->EB){ 
    //l->own=1; l->you=siE-1;
    //while(l->own<=siE){ 
    for(l->own=1,l->you=siE-1;l->own<=siE;l->own++,l->you--){ 
      //l->bit=1; l->ptn=l->TB;
      //while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ 
      for(l->bit=1,l->ptn=l->TB;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->ptn>>=1);
        //l->bit<<=1; l->ptn>>=1; 
      //}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1]--; return; } 
      //if(l->aB[l->own]<l->bit){ break; }
      else if(l->aB[l->own]<l->bit){ break; }
      //l->own++; l->you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(l->own>siE){ 
      l->C4[l->B1]++;
      l->C8[l->B1]--;
      if(DEBUG>0) thMonitor(l,4); 
      return; 
    } 
  }
  //270度回転
  if(l->aB[l->B1]==l->TB){ 
    //l->own=1; l->ptn=l->TB>>1;
    //while(l->own<=siE){ 
    for(l->own=1,l->ptn=l->TB>>1;l->own<=siE;l->own++,l->ptn>>=1){ 
      //l->bit=1; l->you=0;
      //while((l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit)){ 
      for(l->bit=1,l->you=0;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you++);
     // { 
     //   l->bit<<=1; l->you++; 
     // }
      if(l->aB[l->own]>l->bit){ l->C8[l->B1]--; return; } 
      //if(l->aB[l->own]<l->bit){ break; }
      else if(l->aB[l->own]<l->bit){ break; }
      //l->own++; l->ptn>>=1;
    }
  }
}
void backTrack2(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); //配置可能フィールド
  l->bit=0;
  if(y==siE){
    if(bm>0 && (bm&l->LM)==0){  //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      symmetryOps_bm(l);        //対称解除法
    }
  }else{
    if(y<l->B1){                //【枝刈り】上部サイド枝刈り
      bm&=~l->SM; 
    }else if(y==l->B2) {        //【枝刈り】下部サイド枝刈り
      if((down&l->SM)==0){ return; }
      if((down&l->SM)!=l->SM){ bm&=l->SM; }
    }
    while(bm>0) {
      bm^=l->aB[y]=l->bit=-bm&bm;//最も下位の１ビットを抽出
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  }
}
void backTrack1(int y,int left,int down,int right,int bm,local *l){
  bm=l->msk&~(left|down|right); 
  l->bit=0;
  if(y==siE) {
    if(bm>0){//【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      l->aB[y]=bm;
			l->C8[l->B1]++;
      if(DEBUG>0) thMonitor(l,82);
    }
  }else{
    if(y<l->B1) {   //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      bm&=~2; 
    }
    while(bm>0) {   //最も下位の１ビットを抽出
      bm^=l->aB[y]=l->bit=-bm&bm;
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  } 
}
void NQueen(){
  local l;
  l.own=l.ptn=l.you=l.EB=l.SM=l.LM=0;
  l.B1=B1; l.B2=B2; //B1 と B2を初期化
  for(int j=0;j<siE;j++){ l.aB[j]=j; } // aB[]の初期化
	l.C2[l.B1]=l.C4[l.B1]=l.C8[l.B1]=0;	//カウンターの初期化
  l.bit=0 ; l.aB[0]=1; l.msk=(1<<si)-1; l.TB=1<<siE;
  if(l.B1>1 && l.B1<siE) {  //最上段のクイーンが角にある場合の探索
    l.aB[1]=l.bit=(1<<l.B1); //角にクイーンを配置 
    backTrack1(2,(2|l.bit)<<1,(1|l.bit),(l.bit>>1),0,&l);//２行目から探索
  }
  l.EB=(l.TB>>l.B1);
  l.SM=l.LM=(l.TB|1);
  if(l.B1>0&&l.B2<siE&&l.B1<l.B2){  //最上段行のクイーンが角以外にある場合の探索 
    int i;
    for(i=1;i<l.B1;i++){
      l.LM=l.LM|l.LM>>1|l.LM<<1;
    }
    l.aB[0]=l.bit=(1<<l.B1);
    backTrack2(1,l.bit<<1,l.bit,l.bit>>1,0,&l);
    l.EB>>=si;
  }
  printf(":C2:%ld:",l.C2[l.B1]);
  printf("C4:%ld:",l.C4[l.B1]);
  printf("C8:%ld\n",l.C8[l.B1]);
}
int main(int argc, char *argv[]){
  si=atoi(argv[1]);
  siE=si-1;
  B1=atoi(argv[2]);
  B2=atoi(argv[3]);
  printf("si:%d:B1:%d:B2:%d",si,B1,B2);
  NQueen();
  return 0;
}
