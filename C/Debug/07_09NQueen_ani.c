/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

  ９．ビットマップ＋枝刈りと最適化

		コンパイルと実行
		$ make nq9 && ./07_09NQueen


  前章のコードは全ての解を求めた後に、ユニーク解以外の対称解を除去していた
  ある意味、「生成検査法（generate ＆ test）」と同じである
  問題の性質を分析し、バックトラッキング/前方検査法と同じように、無駄な探索を省略することを考える
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
  右半分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合により左右反転することで、
  最小値解とすることが可能だから、中央に配置したものしかユニーク解には成り得ない
  しかし、上辺とその他の辺の中央にクィーンは互いの効きになるので、配置することが出来ない


 枝刈りによる高速化 最上段の行のクイーンの位置に着目
  ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
  右半分だけにクイーンを配置するようにすればよい
  Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
  他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合に
  より左右反転することで、 最小値解とすることが可能だから、中央に配置した
  ものしかユニーク解には成り得ないしかし、上辺とその他の辺の中央にクィーン
  は互いの効きになるので、配置することが出来ない

 最上段行のクイーンが角にある場合の探索
   １行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
   １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
   鏡像についても、主対角線鏡像のみを判定すればよい
   ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい 

  最上段行のクイーンが角以外にある場合の探索
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



 ***********************************************************************
 最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
 ***********************************************************************
 
   次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
 の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
 を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
 ます。突き詰めれば、
 
 [上から２行目のクイーンの位置が右から何番目にあるか]
 [右から２列目のクイーンの位置が上から何番目にあるか]
 

 を比較するだけで判定することができます。この２つの値が同じになることはないからです。
 
       3 0
       ↓↓
---- Q ←0
- Q--- ←3
-----         上から２行目のクイーンの位置が右から４番目にある。
--- Q-         右から２列目のクイーンの位置が上から４番目にある。
-----         しかし、互いの効き筋にあたるのでこれは有り得ない。
 
   結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
 られる解は総てユニーク解であることが保証されます。
 
---- X Q
- Q-- X-
---- X-
---- X-
------
------
 
   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
 ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
 ます。
 
 X X--- Q X X
 X------ X
--------
--------
--------
--------
 X------ X
 X X---- X X
 
   次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
 性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
 て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
 になります。
 
 X X x x x Q X X
 X--- x x x X
 C-- x- x- x
-- x-- x--
- x--- x--
 x---- x- A
 X---- x- X
 X X B-- x X X


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
14:       365596           45752            0.06
15:      2279184          285053            0.34
16:     14772512         1846955            2.20
17:     95815104        11977939           15.16
*/

#include<stdio.h>
#include<time.h>

#define MAX 8 

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
void backTrack2(int si,int msk,int y,int l,int d,int r){
  printf("methodstart:backtrack2\n");
  printf("###y:%d\n",y);
  printf("###l:%d\n",l);
  printf("###d:%d\n",d);
  printf("###r:%d\n",r);
  for(int k=0;k<si;k++){
    printf("###i:%d\n",k);
    printf("###aB[k]:%d\n",aB[k]);
  }
  int bit;
  int bm=msk&~(l|d|r); 
  if(y==si-1){
  printf("if(y==si){\n");
    if(bm>0 && (bm&LASTMASK)==0){ //【枝刈り】最下段枝刈り
      aB[y]=bm;
      symmetryOps_bm(si); //  takakenの移植版の移植版
      //symmetryOps_bm_old(si);// 兄が作成した労作
    }
  }else{
  printf("}else{#y==si\n");
    if(y<BOUND1){             //【枝刈り】上部サイド枝刈り
  printf("y<BOUND1\n");
      bm&=~SIDEMASK; 
    }else if(y==BOUND2) {     //【枝刈り】下部サイド枝刈り
  printf("else if(y==BOUND2)\n");
      if((d&SIDEMASK)==0){ 
        printf("if((d&SIDEMASK)==0){\n");
        return; 
      }
      if((d&SIDEMASK)!=SIDEMASK){ 
        printf("if((d&SIDEMASK)!=SIDEMASK){\n");
        bm&=SIDEMASK; 
      }
    }
  printf("} end else\n");
    while(bm>0) {
  printf("while(bm>0){\n");
      bm^=aB[y]=bit=-bm&bm;
  printf("beforebitmap\n");
  printf("###y:%d\n",y);
  printf("###l:%d\n",l);
  printf("###d:%d\n",d);
  printf("###r:%d\n",r);
  printf("###bm:%d\n",bm);
  for(int k=0;k<si;k++){
    printf("###i:%d\n",k);
    printf("###aB[k]:%d\n",aB[k]);
  }
      backTrack2(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
  printf("afterbitmap\n");
  printf("###y:%d\n",y);
  printf("###l:%d\n",l);
  printf("###d:%d\n",d);
  printf("###r:%d\n",r);
  printf("###bm:%d\n",bm);
  for(int k=0;k<si;k++){
    printf("###i:%d\n",k);
    printf("###aB[k]:%d\n",aB[k]);
  }
    }
  printf("}:end while(bm){\n");
  }
  printf("}:end else\n");
}
void backTrack1(int si,int msk,int y,int l,int d,int r){
  printf("methodstart:backtrack1\n");
  printf("###y:%d\n",y);
  printf("###l:%d\n",l);
  printf("###d:%d\n",d);
  printf("###r:%d\n",r);
  for(int k=0;k<si;k++){
    printf("###i:%d\n",k);
    printf("###aB[k]:%d\n",aB[k]);
  }
  int bm=msk&~(l|d|r); 
  int bit;
  if(y==si-1) {
  printf("if(y==si-1){\n");
    if(bm>0){
  printf("if(bm>0){\n");
      aB[y]=bm;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      C8++;
    }
  }else{
  printf("}else{#y==si-1\n");
    if(y<BOUND1) {   
  printf("if(y<BOUND1){\n");
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
  printf("}#if(y<BOUND1){\n");
    while(bm>0) {
  printf("while(bm>0){\n");
      bm^=aB[y]=bit=-bm&bm;
  printf("beforebitmap\n");
  printf("###y:%d\n",y);
  printf("###l:%d\n",l);
  printf("###d:%d\n",d);
  printf("###r:%d\n",r);
  printf("###bm:%d\n",bm);
  for(int k=0;k<si;k++){
    printf("###i:%d\n",k);
    printf("###aB[k]:%d\n",aB[k]);
  }
      backTrack1(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
  printf("afterbitmap\n");
  printf("###y:%d\n",y);
  printf("###l:%d\n",l);
  printf("###d:%d\n",d);
  printf("###r:%d\n",r);
  printf("###bm:%d\n",bm);
  for(int k=0;k<si;k++){
    printf("###i:%d\n",k);
    printf("###aB[k]:%d\n",aB[k]);
  }
    }
      printf("}:end while(bm){\n");
  } 
    printf("}:end else\n");
}
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
  int min=8; int msk;
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
