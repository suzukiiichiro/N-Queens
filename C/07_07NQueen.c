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
 <>７．ビットマップ+対称解除法          NQueen07() 
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．もっとビットマップ(takaken版)    NQueen10() 
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12() 
   13．マルチスレッド(mutex)            NQueen13() 
   14．マルチスレッド(mutexattr)        NQueen14() 
   15．マルチスレッド(脱mutex COUNT強化)NQueen15() 
   16．アドレスとポインタ(考察１)       NQueen16() 
   17．アドレスとポインタ(考察２)       NQueen17() 
   18．アドレスとポインタ(考察３)       NQueen18()
   19．アドレスとポインタ(考察４)       NQueen19()
   20．アドレスとポインタ(考察５)       NQueen20()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 

 <>７．ビットマップ+対称解除法          NQueen07() N17: 2:24
 *
 *     一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
 *     ８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 * 
 * ■ユニーク解の判定方法
 *   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 * のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 * とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 * 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 * ことを意味しています。
 * 
 *   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 * の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 * どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 * になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 * わりません。
 * 
 *   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 * ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 * いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 * 下の例では要素数が２個のものと８個のものがあります。
 *
 *
 * Ｎ＝５の全解は１０、ユニーク解は２なのです。
 * 
 * グループ１: ユニーク解１つ目
 * - - - Q -   - Q - - -
 * Q - - - -   - - - - Q
 * - - Q - -   - - Q - -
 * - - - - Q   Q - - - -
 * - Q - - -   - - - Q -
 * 
 * グループ２: ユニーク解２つ目
 * - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
 * - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
 * Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
 * - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
 * - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -
 *
 * 
 *   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 * ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 * の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 * をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 * と呼ぶことにします。
 * 
 * - - - - Q   0
 * - - Q - -   2
 * Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
 * - - - Q -   1
 * - Q - - -   3
 * 
 * 
 *   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 * するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 * を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 * とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 * 変換を試みるだけでユニーク解の判定が可能になります。
 *  
 * 

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
15:      2279184          285053      0 00:00:03
16:     14772512         1846955      0 00:00:21
17:     95815104        11977939      0 00:02:30
 */

#include<stdio.h>
#include<time.h>

#define MAX 27

long Total=1 ; //合計解
long Unique=0; //ユニーク解
int aB[MAX];  //チェス盤の横一列
int aT[MAX];
int aS[MAX];
int bit;
int C2=0;int C4=0;int C8=0;

void NQueen(int si,int msk,int y,int l,int d,int r);
void dtob(int score,int si);
void rotate_bitmap(int bf[],int af[],int si);
void vMirror_bitmap(int bf[],int af[],int si);
int rh(int a,int sz);
int intncmp(int lt[],int rt[],int si);
void symmetryOps_bm(int si);
void TimeFormat(clock_t utime,char *form);
long getTotal();
long getUnique();

void NQueen(int si,int msk,int y,int l,int d,int r){
  int bm=msk&~(l|d|r); //配置可能フィールド
  if(y==si&&!bm){
    aB[y]=bm;
    symmetryOps_bm(si);
  }else{
    while(bm) {
      bm^=aB[y]=bit=(-bm&bm); //最も下位の１ビットを抽出
      NQueen(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
    }
  } 
}
int main(void){
  clock_t st; char t[20];
  int min=2; int msk;  //msk:mask
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    Total=0;Unique=0;C2=0;C4=0;C8=0;
    for(int j=0;j<i;j++){ aB[j]=j; }
    msk=(1<<i)-1; // 初期化
    st=clock();
    NQueen(i,msk,0,0,0,0);
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
void dtob(int score,int si) {
  int bit=1; char c[si];
  for (int i=0;i<si;i++) {
    if (score&bit){ c[i]='1'; }else{ c[i]='0'; }
    bit<<=1;
  }
  // 計算結果の表示
  for (int i=si-1;i>=0;i--){ putchar(c[i]); }
  printf("\n");
}
long getUnique(){ 
  return C2+C4+C8;
}
long getTotal(){ 
  return C2*2+C4*4+C8*8;
}
// bf:before af:after
void rotate_bitmap(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
  }
}
void vMirror_bitmap(int bf[],int af[],int si){
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
}
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
int intncmp(int lt[],int rt[],int si){
  int rtn=0;
  for(int k=0;k<si;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
void symmetryOps_bm(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  rotate_bitmap(aT,aS,si);    //時計回りに90度回転
  int k=intncmp(aB,aS,si);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aS,aT,si);  //時計回りに180度回転
    k=intncmp(aB,aT,si);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aT,aS,si);//時計回りに270度回転
      k=intncmp(aB,aS,si);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aS[i]=aB[i];}
  vMirror_bitmap(aS,aT,si);   //垂直反転
  k=intncmp(aB,aT,si);
  if(k>0){ return; }
  if(nEquiv>2){               //-90度回転 対角鏡と同等       
    rotate_bitmap(aT,aS,si);
    k=intncmp(aB,aS,si);
    if(k>0){return;}
    if(nEquiv>4){             //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aB,aT,si);
      if(k>0){ return;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aB,aS,si);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){ C2++; }
  if(nEquiv==4){ C4++; }
  if(nEquiv==8){ C8++; }
}
