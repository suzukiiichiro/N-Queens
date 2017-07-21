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
 <>９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ(takaken版)    NQueen10() 
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12() 
   13．マルチスレッド(mutex)            NQueen13() 
   14．マルチスレッド(mutexattr)        NQueen14() 
   15．マルチスレッド(脱mutex COUNT強化)NQueen15() 
   15t.もっとマルチスレッド(takaken版) NQueen15_t() 
   16．アドレスとポインタ(考察１)       NQueen16() 
   17．アドレスとポインタ(考察２)       NQueen17() 
   18．アドレスとポインタ(考察３)       NQueen18()
   19．アドレスとポインタ(考察４)       NQueen19()
   20．アドレスとポインタ(考察５)       NQueen20()
   21．アドレスとポインタ(考察６)       NQueen21() 
   22．アドレスとポインタ(考察７)       NQueen22() 
   23．アドレスとポインタ(考察８)       NQueen23() 
   24．アドレスとポインタ(完結)         NQueen24() 
   25．最適化 									        NQueen25()
   26．CPUアフィニティ 					        NQueen26()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 

  ９．ビットマップ＋枝刈りと最適化

  前章のコードは全ての解を求めた後に、ユニーク解以外の対称解を除去していた
  ある意味、「生成検査法（generate ＆ test）」と同じである
  問題の性質を分析し、バックトラッキング/前方検査法と同じように、無駄な探索を省略することを考える
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
  右半分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合により左右反転することで、
  最小値解とすることが可能だから、中央に配置したものしかユニーク解には成り得ない
  しかし、上辺とその他の辺の中央にクィーンは互いの効きになるので、配置することが出来ない


  1. １行目角にクイーンがある場合、とそうでない場合で処理を分ける
    １行目かどうかの条件判断はループ外に出してもよい
    処理時間的に有意な差はないので、分かりやすいコードを示した
  2.１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
    １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
    鏡像についても、主対角線鏡像のみを判定すればよい
    ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい

  １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを置くことはできない
  置いた場合、回転・鏡像変換により得られる状態のユニーク判定値が明らかに大きくなる
    ☓☓・・・Ｑ☓☓
    ☓・・・／｜＼☓
    ｃ・・／・｜・rt
    ・・／・・｜・・
    ・／・・・｜・・
    lt・・・・｜・ａ
    ☓・・・・｜・☓
    ☓☓ｂ・・dn☓☓
    
  １行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
  lt, dn, lt 位置は効きチェックで配置不可能となる
  回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、
  90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む

  実行結果
 N:        Total       Unique        hh:mm:ss.ms
 2:            0               0            0.00
 3:            0               0            0.00
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.00
13:        73712            9233            0.01
14:       365596           45752            0.05
15:      2279184          285053            0.35
16:     14772512         1846955            2.28
17:     95815104        11977939           15.77
*/

#include<stdio.h>
#include<time.h>

#define MAX 27

long Total=1 ; //合計解
long Unique=0; //ユニーク解
long C2=0; long C4=0; long C8=0;
int aB[MAX];  //チェス盤の横一列
int aT[MAX];
int aS[MAX];

int MASK;
int BOUND1;
int BOUND2;
int TOPBIT;
int SIZEE;
int SIDEMASK;
int LASTMASK;
int ENDBIT;

void TimeFormat(clock_t utime,char *form);
long getUnique();
long getTotal();
void rotate_bitmap_180(int bf[],int af[],int si);
void rotate_bitmap(int bf[],int af[],int si);
void vMirror_bitmap(int bf[],int af[],int si);
int intncmp(int lt[],int rt[],int si);
int rh(int a,int sz);
void symmetryOps_bm_old(int si);
void symmetryOps_bm(int si);
void backTrack2(int si,int msk,int y,int l,int d,int r);
void backTrack1(int si,int msk,int y,int l,int d,int r);
void NQueen(int si,int msk);

/**********************************************/
/** 対称解除法                               **/
/** ユニーク解から全解への展開               **/
/**********************************************/
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
void symmetryOps_bm(int si){
  int own,ptn,you,bit;
  //90度回転
  if(aB[BOUND2]==1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((aB[you]!=ptn)&&(aB[own]>=bit)){ bit<<=1; you--; }
      if(aB[own]>bit){ return; } if(aB[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ C2++; return; }
  }
  //180度回転
  if(aB[si-1]==ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((aB[you]!=ptn)&&(aB[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aB[own]>bit){ return; } if(aB[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ C4++; return; }
  }
  //270度回転
  if(aB[BOUND1]==TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((aB[you]!=ptn)&&(aB[own]>=bit)){ bit<<=1; you++; }
      if(aB[own]>bit){ return; } if(aB[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  C8++;
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
void backTrack2(int si,int msk,int y,int l,int d,int r){
  int bit;
  int bm=msk&~(l|d|r); 
  if(y==si-1){
    if(bm>0 && (bm&LASTMASK)==0){ //【枝刈り】最下段枝刈り
      aB[y]=bm;
      symmetryOps_bm(si); //  takakenの移植版の移植版
      //symmetryOps_bm_old(si);// 兄が作成した労作
    }
  }else{
    if(y<BOUND1){             //【枝刈り】上部サイド枝刈り
      bm&=~SIDEMASK; 
    }else if(y==BOUND2) {     //【枝刈り】下部サイド枝刈り
      if((d&SIDEMASK)==0){ return; }
      if((d&SIDEMASK)!=SIDEMASK){ bm&=SIDEMASK; }
    }
    while(bm>0) {
      bm^=aB[y]=bit=-bm&bm;
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
  if(y==si-1) {
    if(bm>0){
      aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      C8++;
    }
  }else{
    if(y<BOUND1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bm>0) {
      bm^=aB[y]=bit=-bm&bm;
      backTrack1(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
/**********************************************/
/** 枝刈りによる高速化 
  最上段の行のクイーンの位置に着目 */
/**********************************************/
/**
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
  右半分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合に
  より左右反転することで、 最小値解とすることが可能だから、中央に配置した
  ものしかユニーク解には成り得ないしかし、上辺とその他の辺の中央にクィーン
  は互いの効きになるので、配置することが出来ない
  */
void NQueen(int si,int msk){
  int bit;
  TOPBIT=1<<(si-1);
  aB[0]=1;
  /* 最上段行のクイーンが角にある場合の探索 */
  for(BOUND1=2;BOUND1<si-1;BOUND1++){
    // 角にクイーンを配置 
    aB[1]=bit=(1<<BOUND1); 
    //２行目から探索
    backTrack1(si,msk,2,(2|bit)<<1,(1|bit),(bit>>1)); 
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  for(BOUND1=1,BOUND2=si-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aB[0]=bit=(1<<BOUND1);
    backTrack2(si,msk,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
int main(void){
  clock_t st; char t[20];
  int min=2; int msk;
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    Total=0;Unique=0;C2=C4=C8=0;
    for(int j=0;j<i;j++){ aB[j]=j; }
    msk=(1<<i)-1; // 初期化
    st=clock();
    NQueen(i,msk);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
  } 
  return 0;
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
  return C2+C4+C8;
}
long getTotal(){ 
  return C2*2+C4*4+C8*8;
}
// bf:before af:after
void rotate_bitmap_180(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[si-i-1]>>j)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
  }
}
void rotate_bitmap(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
  }
}
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
void vMirror_bitmap(int bf[],int af[],int si){
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
}
int intncmp(int lt[],int rt[],int si){
  int rtn=0;
  for(int k=0;k<si;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
void symmetryOps_bm_old(int si){
  int aT[si];
  int aS[si];
  int k;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  // 1行目のクイーンのある位置を基準にこれを 90度回転 180度回転 270度回転させた時に重なるかどうか判定する
  // Aの場合 １行目のクイーンのある場所の左端からの列数が1の時（９０度回転したところ）
  if(aB[BOUND2]==1){
    rotate_bitmap(aT,aS,si);  //時計回りに90度回転
    k=intncmp(aB,aS,si);
    if(k>0)return;
    if(k==0){ C2++; return;}
  }
  // Bの場合 1番下の行の現在の左端から2番目が1の場合 180度回転
  if(aB[si-1]==ENDBIT){
    //aB[BOUND2]==1のif 文に入っていない場合は90度回転させてあげる
    //   if(aB[BOUND2]!=1){
    //     rotate_bitmap(aT,aS);  //時計回りに90度回転
    //   }
    //rotate_bitmap(aS,aT);    //時計回りに180度回転
    rotate_bitmap_180(aB,aT,si);    //時計回りに180度回転
    k=intncmp(aB,aT,si);
    if(k>0)return;
    if(k==0){ C4++; return;}
  }
  //Cの場合 1行目のクイーンのある位置の右端からの列数の行の左端が1の時 270度回転
  if(aB[BOUND1]==TOPBIT){
    //aB[BOUND2]==1のif 文に入っていない場合は90度回転させてあげる
    if(aB[BOUND2]!=1){
      rotate_bitmap(aT,aS,si);  //時計回りに90度回転
    }
    //aB[si-1]!=ENDBITのif 文に入っていない場合は180度回転させてあげる
    if(aB[si-1]!=ENDBIT){
      rotate_bitmap(aS,aT,si);//時計回りに180度回転
    }
    rotate_bitmap(aT,aS,si);//時計回りに270度回転
    k=intncmp(aB,aS,si);
    if(k>0){ return;}
  }
  C8++;
}
