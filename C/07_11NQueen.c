/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() 
   ４．対称解除法(回転と斜軸）          NQueen04() 
   ５．枝刈りと最適化                   NQueen05() 
   ６．ビットマップ                     NQueen06() 
   ７．ビットマップ+対称解除法          NQueen07() 
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ(takaken版)    NQueen10() 
 <>11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12() 
   13．マルチスレッド(join)             NQueen13() 
   14．マルチスレッド(mutex)            NQueen14() 
   15．マルチスレッド(アトミック対応)   NQueen15() 
   16．アドレスとポインタ               NQueen16() 
   17．アドレスとポインタ(脱構造体)     NQueen17() 
   18．アドレスとポインタ(脱配列)       NQueen18()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 

  11.マルチスレッド（構造体）

  実行結果
 N:        Total       Unique        dd:hh:mm:ss
 2:            0               0      0 00:00:00
 3:            0               0      0 00:00:00
 4:            2               1      0 00:00:00
 5:           10               2      0 00:00:00
 6:            4               1      0 00:00:00
 7:           40               6      0 00:00:00
 8:           92              12      0 00:00:00
 9:          352              46      0 00:00:00
10:          724              92      0 00:00:00
11:         2680             341      0 00:00:00
12:        14200            1787      0 00:00:00
13:        73712            9233      0 00:00:00
14:       365596           45752      0 00:00:00
15:      2279184          285053      0 00:00:00
16:     14772512         1846955      0 00:00:02
17:     95815104        11977939      0 00:00:14
*/

#include<stdio.h>
#include<time.h>

#define MAXSIZE 27

// pthreadはパラメータを１つしか渡せないので構造体に格納
typedef struct {
  int BOUND1;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int SIDEMASK;
  int LASTMASK;
  long C2;
  long C4;
  long C8;
  int aB[MAXSIZE];
}CLASS, *Class;
CLASS C; //構造体

void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void backTrack2(int si,int msk,int y,int l,int d,int r);
void backTrack1(int si,int msk,int y,int left,int down,int right);
void run(int si,int msk);
void NQueenThread(int si,int msk);

/**********************************************/
/** 対称解除法                               **/
/** ユニーク解から全解への展開               **/
/**********************************************/
/**
ひとつの解には、盤面を90度・180度・270度回転、及びそれらの鏡像の合計8個の対称解が存在する

    １２ ４１ ３４ ２３
    ４３ ３２ ２１ １４

    ２１ １４ ４３ ３２
    ３４ ２３ １２ ４１
    
上図左上がユニーク解。
1行目はユニーク解を90度、180度、270度回転したもの
2行目は1行目のそれぞれを左右反転したもの。
2行目はユニーク解を左右反転、対角反転、上下反転、逆対角反転したものとも解釈可 
ただし、 回転・線対称な解もある
**/
/**
クイーンが右上角にあるユニーク解を考えます。
斜軸で反転したパターンがオリジナルと同型になることは有り得ないことと(×２)、
右上角のクイーンを他の３つの角に写像させることができるので(×４)、
このユニーク解が属するグループの要素数は必ず８個(＝２×４)になります。

(1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルから180度回転)
　　させても、さらに90度回転(オリジナルから270度回転)させてもオリジナルと同型になる。 
(2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナルとは異なる。
　　ただし、180度回転させた場合はオリジナルと同型になることも有り得る。

　(1)に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを加えて
２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、180度回転させ
て同型になる場合は４個(左右反転×縦横回転)、そして180度回転させてもオリジナルと異なる
場合は８個になります。(左右反転×縦横回転×上下反転)
*/
void symmetryOps_bm(int si){
  int own,ptn,you,bit;
  //90度回転
  if(C.aB[C.BOUND2]==1){ own=1; ptn=2;
    while(own<=(si-1)){ bit=1; you=(si-1);
      while((C.aB[you]!=ptn)&&(C.aB[own]>=bit)){ bit<<=1; you--; }
      if(C.aB[own]>bit){ return; } if(C.aB[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>(si-1)){ C.C2++; return; }
  }
  //180度回転
  if(C.aB[(si-1)]==C.ENDBIT){ own=1; you=(si-1)-1;
    while(own<=(si-1)){ bit=1; ptn=C.TOPBIT;
      while((C.aB[you]!=ptn)&&(C.aB[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(C.aB[own]>bit){ return; } if(C.aB[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>(si-1)){ C.C4++; return; }
  }
  //270度回転
  if(C.aB[C.BOUND1]==C.TOPBIT){ own=1; ptn=C.TOPBIT>>1;
    while(own<=(si-1)){ bit=1; you=0;
      while((C.aB[you]!=ptn)&&(C.aB[own]>=bit)){ bit<<=1; you++; }
      if(C.aB[own]>bit){ return; } if(C.aB[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  C.C8++;
}
/**********************************************/
/* 最上段行のクイーンが角以外にある場合の探索 */
/**********************************************/
/**
１行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを
置くことはできない
１行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
lt, dn, lt 位置は効きチェックで配置不可能となる
回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、 
90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む

【枝刈り図】
  x x - - - Q x x    
  x - - - / | ＼x    
  c - - / - | -rt    
  - - / - - | - -    
  - / - - - | - -    
  lt- - - - | - a    
  x - - - - | - x    
  x x b - - dnx x    
*/
//void backTrack2(int y,int left,int down,int right){
void backTrack2(int si,int msk,int y,int l,int d,int r){
  int bit=0;
  int bm=msk&~(l|d|r); 
  if(y==(si-1)){
    if(bm>0&&(bm&C.LASTMASK)==0){ //【枝刈り】最下段枝刈り
      C.aB[y]=bm;
      symmetryOps_bm(si); //  takakenの移植版の移植版
    }
  }else{
    if(y<C.BOUND1){             //【枝刈り】上部サイド枝刈り
      bm&=~C.SIDEMASK; 
      // bm|=SIDEMASK; 
      // bm^=SIDEMASK;(bm&=~SIDEMASKと同等)
    }else if(y==C.BOUND2) {     //【枝刈り】下部サイド枝刈り
      if((d&C.SIDEMASK)==0){ return; }
      if((d&C.SIDEMASK)!=C.SIDEMASK){ bm&=C.SIDEMASK; }
    }
    while(bm>0) {
      bm^=C.aB[y]=bit=-bm&bm;
      //backTrack2(y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
      backTrack2(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  }
}
/**********************************************/
/* 最上段行のクイーンが角にある場合の探索     */
/**********************************************/
/* 
１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
１行目角にクイーンがある場合、他の角にクイーンを配置することは不可
鏡像についても、主対角線鏡像のみを判定すればよい
２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい 
*/
void backTrack1(int si,int msk,int y,int l,int d,int r){
  int bm=msk&~(l|d|r); 
  int bit;
  if(y==(si-1)) {
    if(bm>0){
      C.aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      C.C8++;
    }
  }else{
    if(y<C.BOUND1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bm>0) {
      bm^=C.aB[y]=bit=-bm&bm;
      backTrack1(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
void run(int si,int msk){
  int bit ;
  C.aB[0]=1;
  C.TOPBIT=1<<(si-1);
  // 最上段のクイーンが角にある場合の探索
  if(C.BOUND1>1&&C.BOUND1<(si-1)) { 
    if(C.BOUND1<(si-1)) {
      C.aB[1]=bit=(1<<C.BOUND1);
      backTrack1(si,msk,2,(2|bit)<<1,(1|bit),(bit>>1));
    }
  }
  C.ENDBIT=(C.TOPBIT>>C.BOUND1);
  C.SIDEMASK=C.LASTMASK=(C.TOPBIT|1);
  // 最上段のクイーンが角以外にある場合の探索
  if(C.BOUND1>0&&C.BOUND2<si-1&&C.BOUND1<C.BOUND2){ 
    for(int i=1;i<C.BOUND1;i++){
      C.LASTMASK=C.LASTMASK|C.LASTMASK>>1|C.LASTMASK<<1;
    }
    if(C.BOUND1<C.BOUND2) {
      C.aB[0]=bit=(1<<C.BOUND1);
      backTrack2(si,msk,1,bit<<1,bit,bit>>1);
    }
    C.ENDBIT>>=si;
  }
}
void NQueenThread(int si,int msk){
  for(C.BOUND1=si-1,C.BOUND2=0;C.BOUND2<si-1;C.BOUND1--,C.BOUND2++){
     run(si,msk);
  }
}
int main(void){
  clock_t st; char t[20];
  int min=2;int msk;
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=min;i<=MAXSIZE;i++){
    C.C2=C.C4=C.C8=0;
    for(int j=0;j<i;j++){ C.aB[j]=j; }
    msk=(1<<i)-1; // 初期化
    st=clock();
    NQueenThread(i,msk);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
  } 
}
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
  if (dd) sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
  else if (hh) sprintf(form, "     %2d:%02d:%05.2f",hh,mm,ss);
  else if (mm) sprintf(form, "        %2d:%05.2f",mm,ss);
  else sprintf(form, "           %5.2f",ss);
}
long getUnique(){ 
  return C.C2+C.C4+C.C8;
}
long getTotal(){ 
  return C.C2*2+C.C4*4+C.C8*8;
}
