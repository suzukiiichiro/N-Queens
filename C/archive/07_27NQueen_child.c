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
void dtob(int score,int si) {
  int bit=1; char c[si];
  for (int i=0;i<si;i++) {
    if (score&bit){ c[i]='1'; }else{ c[i]='0'; }
    bit<<=1;
  }
  for (int i=si-1;i>=0;i--){ putchar(c[i]); }
  printf("\n");
}
void dtoq(local *l,int si) {
  for (int y=0;y<si;y++) {
    int bit=1; char c[si];
    for (int i=0;i<si;i++) {
      if (l->aB[y]&bit){ c[i]='Q'; }else{ c[i]='-'; }
        bit<<=1;
    }
    for (int i=si-1;i>=0;i--){ putchar(c[i]); }
    printf("\n");
  }
  printf("\n");
}


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
  /**
  printf("\033[G");
  if(i==2){
    printf("\rN:%2d C2[%c] C4[ ] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C2[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  else if(i==4){
    printf("\rN:%2d C2[ ] C4[%c] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C4[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  else if(i==8){
    printf("\rN:%2d C2[ ] C4[ ] C8[%c] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  else if(i==82){ 
    printf("\rN:%2d C2[ ] C4[ ] C8[ ] C8BT[%c] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1][l->BK]%spl],l->B1,l->B2);
  }
  */
  printf("\033[G");
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
    putchar('\n');
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
  //printf("####si:%d:B1:%d:y:%d:bk2\n",si,l->B1,y);
  //printf("msk\n");
  //dtob(l->msk,si);
  //printf("left\n");
  //dtob(left,si);
  //printf("down\n");
  //dtob(down,si);
  //printf("right\n");
  //dtob(right,si);
  //printf("bbm\n");
  //dtob(bm,si);
  bm=l->msk&~(left|down|right); 
  //printf("abm:%d\n",bm);
  //dtob(bm,si);
  l->bit=0;
  if(y==siE){
    //printf("y==siE\n");
    //printf("bm:%d\n",bm);
    //dtob(bm,si);
    //printf("LM:%d\n",l->LM);
    //dtob(l->LM,si);
    //printf("bm&l->LM:%d\n",bm&l->LM);
    //dtob(bm&l->LM,si);
    //(bm&l->LM)==0
    //最下段でLMにひっかかるものはここで刈り取られる
    //最下段はLMに当たる場所にクイーンはおけない
    //両端どちらかが1
    //この場合はOK
    //bm      :00100000
    //LM      :11000011
    //bm&l->LM:00000000
    //この場合は刈り取られる
    //bm      :00000010
    //LM      :11000011
    //bm&l->LM:00000010
    if(bm>0 && (bm&l->LM)==0){  //【枝刈り】最下段枝刈り
      l->aB[y]=bm;
      symmetryOps_bm(l);        //対称解除法
    }else{
      printf("y==siE:bm=0\n");
      thMonitor(l,82); 
      printf("left:%d\n",left);
      dtob(left,si);
      printf("down:%d\n",down);
      dtob(down,si);
      printf("right:%d\n",right);
      dtob(right,si);
      printf("l->msk&~(left|down|right):%d\n",l->msk&~(left|down|right));
      dtob(l->msk&~(left|down|right),si);
    }
  }else{
    if(y<l->B1){                //【枝刈り】上部サイド枝刈り
      //printf("y<l->B1\n");
      //printf("y:%d\n",y);
      //printf("B1:%d\n",l->B1);
      //printf("bbm\n");
      //dtob(bm,si);
      //printf("SM\n");
      //dtob(l->SM,si);
      bm&=~l->SM; 
      //SMは左右両端が1 10000001
      //左右両端を刈り込む
      //bm:11110001
      //SM:10000001
      //bm:01110000
      //printf("abm:%d\n",bm);
      //dtob(bm,si);
    }else if(y==l->B2) {        //【枝刈り】下部サイド枝刈り
      //printf("y==l->B2\n");
      //printf("y:%d\n",y);
      //printf("B2:%d\n",l->B2);
      //printf("down:%d\n",down);
      //dtob(down,si);
      //printf("SM:%d\n",l->SM);
      //dtob(l->SM,si);
      //printf("down&SM\n");
      //dtob(down&l->SM,si);
      //downの両端が0の場合にdown&SM=0になる
      //down   :10011110
      //SM     :10000001
      //down&SM:10000000
      //down   :01011110
      //SM     :10000001
      //down&SM:00000000
      if((down&l->SM)==0){ 
        //printf("(down&l->SM)==0\n");
        printf("y==B2:down&SM==0\n");
        thMonitor(l,82); 
        printf("left:%d\n",left);
        dtob(left,si);
        printf("down:%d\n",down);
        dtob(down,si);
        printf("right:%d\n",right);
        dtob(right,si);
        printf("l->msk&~(left|down|right):%d\n",l->msk&~(left|down|right));
        dtob(l->msk&~(left|down|right),si);
        return; 
      }
      //printf("down&l->SM\n");
      //dtob(down&l->SM,si);
      //printf("SM\n");
      //dtob(l->SM,si);
      //(down&l->SM)!=l->SM
      //両端どちらも1の場合は(down&l->SM)==l->SM
      if((down&l->SM)!=l->SM){ 
        //printf("(down&l->SM)!=l->SM\n");
        //printf("down:%d\n",down);
        //dtob(down,si);
        //printf("SM:%d\n",l->SM);
        //dtob(l->SM,si);
        //printf("bbm\n");
        //dtob(bm,si);
        //両端の1だけ残す
        //bm:00000001
        //SM:10000001
        //bm:00000001
        bm&=l->SM; 
        //printf("abm:%d\n",bm);
        //dtob(bm,si);
      }
    }
    while(bm>0) {
      //printf("bm>0:bm^=l->aB[y]=l->bit=-bm&bm\n");
      //printf("bbm\n");
      //dtob(bm,si);
      bm^=l->aB[y]=l->bit=-bm&bm;//最も下位の１ビットを抽出
      //printf("abm:%d\n",bm);
      //dtob(bm,si);
      //printf("y:%d:aB[y]:%d\n",y,l->aB[y]);
      //dtob(l->aB[y],si);
      //printf("left:%d\n",left);
      //dtob(left,si);
      //printf("bit:%d\n",l->bit);
      //dtob(l->bit,si);
      //printf("left<<1:%d\n",(left|l->bit)<<1);
      //dtob((left|l->bit)<<1,si);
      //printf("down:%d\n",down);
      //dtob(down,si);
      //printf("down|l->bit:%d\n",down|l->bit);
      //dtob(down|l->bit,si);
      //printf("bit:%d\n",l->bit);
      //dtob(l->bit,si);
      //printf("right:%d\n",right);
      //dtob(right,si);
      //printf("bit:%d\n",l->bit);
      //dtob(l->bit,si);
      //printf("right>>1:%d\n",(right|l->bit)>>1);
      //dtob((right|l->bit)>>1,si);
      //printf("##recur\n");
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
    printf("y<siE:bm==0\n");
    printf("bm:%d\n",bm);
    thMonitor(l,82); 
    printf("left:%d\n",left);
    dtob(left,si);
    printf("down:%d\n",down);
    dtob(down,si);
    printf("right:%d\n",right);
    dtob(right,si);
    printf("l->msk&~(left|down|right):%d\n",l->msk&~(left|down|right));
    dtob(l->msk&~(left|down|right),si);
  }
}
void backTrack1(int y,int left,int down,int right,int bm,local *l){
  //printf("###si:%d:B1:%d:y:%d:bk1\n",si,l->B1,y);
  //printf("msk\n");
  //dtob(l->msk,si);
  //printf("left\n");
  //dtob(left,si);
  //printf("down\n");
  //dtob(down,si);
  //printf("right\n");
  //dtob(right,si);
  //printf("bbm\n");
  //dtob(bm,si);
  bm=l->msk&~(left|down|right); 
  //bmはクイーンが置ける場所
  //l->msk はsi分1が並んでいる
  //そこから引数に渡されてきたleft,right,downを取り除く。
  //msk
  //11111111
  //left
  //00011000
  //down
  //00001010
  //right
  //00000100
  //bmp
  //11100001
  //printf("abm:%d\n",bm);
  //dtob(bm,si);
  l->bit=0;
  if(y==siE) {
  //printf("y==siE:y:%d:siE:%d\n",y,siE);
  //printf("bm");
  //dtob(bm,si);
  //yが1番下に来たら
    if(bm>0){//【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      //printf("bm:%d\n",bm);
      //dtob(bm,si);
      //1番下の行にクイーンを置けるか判定する
      //bmは残り1個しか残っていないので bm>0かどうかだけ判定し
      //0だったら配置する場所がないので抜ける
      //0より大きければ最下位のビットを抽出するまでものくその値がaB[y]になる
      //bm:   00001000
      //l->aB:00001000
      l->aB[y]=bm;
      //backtrack1の場合は無条件でC8カウントアップ
			l->C8[l->B1]++;
      if(DEBUG>0) thMonitor(l,82);
    }else{
    //bmが0なら置く場所がないので終了
     printf("y==siE:bm=0\n"); 
     thMonitor(l,82); 
     printf("left:%d\n",left);
     dtob(left,si);
     printf("down:%d\n",down);
     dtob(down,si);
     printf("right:%d\n",right);
     dtob(right,si);
     printf("l->msk&~(left|down|right):%d\n",l->msk&~(left|down|right));
     dtob(l->msk&~(left|down|right),si);
    }
  }else{
    if(y<l->B1) {   //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      //printf("y<l->B1\n");
      //printf("y:%d\n",y);
      //printf("B1:%d\n",l->B1);
      //printf("bbm\n");
      //dtob(bm,si);
      //backtrack1ではy<B1の間は右から2個目にクイーンを配置しない。
      //これでユニーク解であることが保証される
      //bm:10001010
      // 2:00000010
      //bm:10001000 
      bm&=~2; 
      //printf("abm:%d\n",bm);
      //dtob(bm,si);
    }
    while(bm>0) {   //最も下位の１ビットを抽出
      //bmが0になると抜ける
      //最も下位の1をとってaB[y],l->bitに設定する
      //bmが0になるとクイーンを置ける可能性がある場所がなくなるので抜ける
      //yが最後まで行っていなくてもbmが0になれば抜ける
      //printf("bm>0:bm^=l->aB[y]=l->bit=-bm&bm\n");
      //printf("bbm\n");
      //dtob(bm,si);
      bm^=l->aB[y]=l->bit=-bm&bm;
      //最も下位の１ビットを抽出
      //bmの中で1番桁数が少ない1を0にする
      //aB[y],l->bitにその値を設定する
      //11100001
      //この場合1番桁数の低い右端の1が選択される
      //aB[y]
      //00000001
      //bm
      //11100000
      //printf("abm:%d\n",bm);
      //dtob(bm,si);
      //printf("y:%d:aB[y]:%d\n",y,l->aB[y]);
      //dtob(l->aB[y],si);
      //printf("left:%d\n",left);
      //dtob(left,si);
      //printf("bit:%d\n",l->bit);
      //dtob(l->bit,si);
      //printf("left<<1:%d\n",(left|l->bit)<<1);
      //dtob((left|l->bit)<<1,si);
      //printf("down:%d\n",down);
      //dtob(down,si);
      //printf("bit:%d\n",l->bit);
      //dtob(l->bit,si);
      //printf("down|l->bit:%d\n",down|l->bit);
      //dtob(down|l->bit,si);
      //printf("right:%d\n",right);
      //dtob(right,si);
      //printf("bit:%d\n",l->bit);
      //dtob(l->bit,si);
      //printf("right>>1:%d\n",(right|l->bit)>>1);
      //dtob((right|l->bit)>>1,si);
      //printf("##recur\n");
      //次のbacktrackに渡すleft,down,rightを設定する
      //left,down,rightは、y1から蓄積されていく
      //left はleft(今までのleftライン)とl->bit(今回選択されたクイーンの位置)を左に1ビットシフト
      //left        00110010
      //l->bit      00000100
      //left|l->bit 00110110
      //1bit左シフト01101100
      //downはdown(今までのdownライン) と l->bit(今回選択されたクイーンの位置)
      //down        00001011
      //l->bit      00000100
      //down|l->bit 00001111
      //rightはright(今までのrightライン)とl->bit(今回選択されたクイーンの位置)を右に1ビットシフト
      //right       00000010
      //l->bit      00000100
      //right|l->bit00000110
      //1bit右シフト00000011
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
    printf("y<siE:bm=0\n");
    thMonitor(l,82); 
    printf("bm:%d\n",bm);
    printf("left:%d\n",left);
    dtob(left,si);
    printf("down:%d\n",down);
    dtob(down,si);
    printf("right:%d\n",right);
    dtob(right,si);
    printf("l->msk&~(left|down|right):%d\n",l->msk&~(left|down|right));
    dtob(l->msk&~(left|down|right),si);
  } 
}
void NQueen(){
  local l;
  l.own=l.ptn=l.you=l.EB=l.SM=l.LM=0;
  l.B1=B1; l.B2=B2; //B1 と B2を初期化
  for(int j=0;j<siE;j++){ l.aB[j]=j; } // aB[]の初期化
	l.C2[l.B1]=l.C4[l.B1]=l.C8[l.B1]=0;	//カウンターの初期化
  l.bit=0 ; l.aB[0]=1; l.msk=(1<<si)-1; l.TB=1<<siE;
  //printf("msk:%d:si:%d\n",l.msk,si);
  //dtob(l.msk,si);
  if(l.B1>1 && l.B1<siE) {  //最上段のクイーンが角にある場合の探索
    l.aB[1]=l.bit=(1<<l.B1); //角にクイーンを配置 
    backTrack1(2,(2|l.bit)<<1,(1|l.bit),(l.bit>>1),0,&l);//２行目から探索
  }
  l.EB=(l.TB>>l.B1);
  l.SM=l.LM=(l.TB|1);
  //printf("EB:%d\n",l.EB);
  //dtob(l.EB,si);
  //printf("TB:%d\n",l.TB);
  //dtob(l.TB,si);
  //printf("SM:%d\n",l.SM);
  //dtob(l.SM,si);
  //printf("LM:%d\n",l.LM);
  //dtob(l.LM,si);
  if(l.B1>0&&l.B2<siE&&l.B1<l.B2){  //最上段行のクイーンが角以外にある場合の探索 
    int i;
    for(i=1;i<l.B1;i++){
      l.LM=l.LM|l.LM>>1|l.LM<<1;
    }
    l.aB[0]=l.bit=(1<<l.B1);
    backTrack2(1,l.bit<<1,l.bit,l.bit>>1,0,&l);
    l.EB>>=si;
  }
  //printf(":C2:%ld:",l.C2[l.B1]);
  //printf("C4:%ld:",l.C4[l.B1]);
  //printf("C8:%ld\n",l.C8[l.B1]);
}
int main(int argc, char *argv[]){
  si=atoi(argv[1]);
  siE=si-1;
  B1=atoi(argv[2]);
  B2=atoi(argv[3]);
  //printf("########si:%d:B1:%d:B2:%d\n",si,B1,B2);
  NQueen();
  return 0;
}
