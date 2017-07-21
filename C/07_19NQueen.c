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
   11．マルチスレッド(構造体)           NQueen11() 
   12．マルチスレッド(pthread)          NQueen12() 
   13．マルチスレッド(join)             NQueen13() 
   14．マルチスレッド(mutex)            NQueen14() 
   15．マルチスレッド(アトミック対応)   NQueen15() 
   16．アドレスとポインタ               NQueen16() 
   17．アドレスとポインタ(脱構造体)     NQueen17() 
   18．アドレスとポインタ(脱配列)       NQueen18()
 <>19．アドレスとポインタ(考察４)       NQueen19() 
   20．アドレスとポインタ(考察５)       NQueen20() 
   21．アドレスとポインタ(考察６)       NQueen21() 
   22．アドレスとポインタ(考察７)       NQueen22() 
   23．アドレスとポインタ(考察８)       NQueen23() 
   24．アドレスとポインタ(完結)         NQueen24() 
   25．最適化 									        NQueen25()
   26．CPUアフィニティ 					        NQueen26()

 # Java/C/Lua/Bash版
 # https://github.com/suzukiiichiro/N-Queen
 


 <>19．アドレスとポインタ(考察４)
 実行結果

1,構造体のみ追加
void backTrack2(int y,int left,int down,int right,struct local *l,int *si,
     int *bo,int *bo2,int *ma,int *sm,int *lm,int *to,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.15
16:        14772512          1846955        0000:00:00.99
17:        95815104         11977939        0000:00:06.12

2,SIZEEを移動
void backTrack2(int y,int left,int down,int right,struct local *l,
     int *bo,int *bo2,int *ma,int *sm,int *lm,int *to,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.16
16:        14772512          1846955        0000:00:00.93
17:        95815104         11977939        0000:00:05.74

3,BOUND1を移動
void backTrack2(int y,int left,int down,int right,struct local *l,
     int *bo2,int *ma,int *sm,int *lm,int *to,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.14
16:        14772512          1846955        0000:00:00.88
17:        95815104         11977939        0000:00:05.55

4,BOUND2を移動
void backTrack2(int y,int left,int down,int right,struct local *l,
     int *ma,int *sm,int *lm,int *to,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.13
16:        14772512          1846955        0000:00:00.80
17:        95815104         11977939        0000:00:05.23

5,MASKを移動
void backTrack2(int y,int left,int down,int right,struct local *l,
     int *sm,int *lm,int *to,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.15
16:        14772512          1846955        0000:00:00.79
17:        95815104         11977939        0000:00:05.10

6,SIDEMASKを移動
void backTrack2(int y,int left,int down,int right,struct local *l,
     int *lm,int *to,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.12
16:        14772512          1846955        0000:00:00.82
17:        95815104         11977939        0000:00:05.10
18:       666090624         83263591        0000:00:35.18

7,LASTMASKを移動
void backTrack2(int y,int left,int down,int right,struct local *l,
     int *to,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.11
16:        14772512          1846955        0000:00:00.74
17:        95815104         11977939        0000:00:04.71

8,TOPBITを移動
void backTrack2(int y,int left,int down,int right,struct local *l,int *en,int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.14
16:        14772512          1846955        0000:00:00.70
17:        95815104         11977939        0000:00:04.50

9,ENDBITを移動
void backTrack2(int y,int left,int down,int right,struct local *l,
     int *p,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.10
16:        14772512          1846955        0000:00:00.65
17:        95815104         11977939        0000:00:04.31

10,aBoardを移動
void backTrack2(int y,int left,int down,int right,struct local *l,long *c2,long *c4,long *c8){
15:         2279184           285053        0000:00:00.10
16:        14772512          1846955        0000:00:00.58
17:        95815104         11977939        0000:00:03.69

10,COUNT2を移動
void backTrack2(int y,int left,int down,int right,struct local *l,long *c4,long *c8){
15:         2279184           285053        0000:00:00.11
16:        14772512          1846955        0000:00:00.56
17:        95815104         11977939        0000:00:03.68

11,COUNT4を移動
void backTrack2(int y,int left,int down,int right,struct local *l, long *c8){
15:         2279184           285053        0000:00:00.09
16:        14772512          1846955        0000:00:00.56
17:        95815104         11977939        0000:00:03.65

12,COUNT8を移動
void backTrack2(int y,int left,int down,int right,struct local *l,){
15:         2279184           285053        0000:00:00.11
16:        14772512          1846955        0000:00:00.65
17:        95815104         11977939        0000:00:04.21

13,COUNT2,COUNT4,COUNT8を外出しにする。SIDEMASKを変数に入れて使う
15:         2279184           285053        0000:00:00.10
16:        14772512          1846955        0000:00:00.57
17:        95815104         11977939        0000:00:03.66
*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <math.h>
#include "pthread.h"
#include <sys/time.h>
#define MAXSIZE 27

// pthreadはパラメータを１つしか渡せないので構造体に格納
/** スレッドローカル構造体 */
struct local{
  int BOUND1;
  int BOUND2;
  int TOPBIT;
  int ENDBIT;
  int MASK;
  int SIDEMASK;
  int LASTMASK;
  int aBoard[MAXSIZE];
  int SIZE;
  int SIZEE;
  long COUNT2;
  long COUNT4;
  long COUNT8;
};
//グローバル構造体
typedef struct {
  int SIZE; //SIZEはスレッドローカルにコピーします。
  int SIZEE;//SIZEEはスレッドローカルにコピーします。
  long lTotal;
  long lUnique;
}GCLASS, *GClass;
GCLASS G; //グローバル構造体

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
void symmetryOps_bitmap(struct local *l,long *c2,long *c4,long *c8){
  int own,ptn,you,bit;
  //90度回転
  if(l->aBoard[l->BOUND2]==1){ own=1; ptn=2;
    while(own<=l->SIZEE){ bit=1; you=l->SIZEE;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you--; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; } own++; ptn<<=1; }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>l->SIZEE){ (*c2)++; return; } }
  //180度回転
  if(l->aBoard[l->SIZEE]==l->ENDBIT){ own=1; you=l->SIZEE-1;
    while(own<=l->SIZEE){ bit=1; ptn=l->TOPBIT;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; } own++; you--; }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>l->SIZEE){ (*c4)++; return; } }
  //270度回転
  if(l->aBoard[l->BOUND1]==l->TOPBIT){ own=1; ptn=l->TOPBIT>>1;
    while(own<=l->SIZEE){ bit=1; you=0;
      while((l->aBoard[you]!=ptn)&&(l->aBoard[own]>=bit)){ bit<<=1; you++; }
      if(l->aBoard[own]>bit){ return; } if(l->aBoard[own]<bit){ break; } own++; ptn>>=1; } }
  (*c8)++;
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
void backTrack2(int y,int left,int down,int right,struct local *l,
     long *c2,long *c4,long *c8){
  int bit=0; int bitmap=l->MASK&~(left|down|right); //配置可能フィールド
  int sm=l->SIDEMASK;
  if(y==l->SIZEE){
    if(bitmap!=0){ //【枝刈り】最下段枝刈り
      if( (bitmap&l->LASTMASK)==0){ 
        l->aBoard[y]=bitmap;
        //対称解除法
        symmetryOps_bitmap(l,c2,c4,c8); } }
  }else{
    if(y<l->BOUND1){ //【枝刈り】上部サイド枝刈り
      bitmap&=~sm; 
    }else if(y==l->BOUND2) { //【枝刈り】下部サイド枝刈り
      if((down&sm)==0){ return; }
      if((down&sm)!=sm){ bitmap&=sm; } }
    while(bitmap!=0) { //最も下位の１ビットを抽出
      bitmap^=l->aBoard[y]=bit=-bitmap&bitmap;
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,
                                  l,c2,c4,c8); } }
}
/**********************************************/
/*  枝刈りと最適化                            */
/**********************************************/
/**
 *
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
*/

/**********************************************/
/* 最上段行のクイーンが角にある場合の探索     */
/**********************************************/
/* 
１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
１行目角にクイーンがある場合、他の角にクイーンを配置することは不可
鏡像についても、主対角線鏡像のみを判定すればよい
２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい 
*/
void backTrack1(int y,int left,int down,int right,struct local *l,long *c8){
  int bit; int bitmap=l->MASK&~(left|down|right);  //配置可能フィールド
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(y==l->SIZEE) { if(bitmap!=0){ l->aBoard[y]=bitmap; (*c8)++; }
  }else{
    //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
    // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
    // bitmap|=2; // bitmap^=2; //(bitmap&=~2と同等)
    if(y<l->BOUND1) { bitmap&=~2; }
    //最も下位の１ビットを抽出
    while(bitmap!=0) {
      bitmap^=l->aBoard[y]=bit=(-bitmap&bitmap);
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,l,c8); } } 
}
/**
  クイーンの場所で分岐
 *
 * ■ユニーク解の個数を求める
 *   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 * ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 * 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 * うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 * れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 * 央になければならず、それは互いの効き筋にあたるので有り得ません。
 *
 *
 * ***********************************************************************
 * 最上段の行のクイーンの位置は中央を除く右側の領域に限定されます。(ただし、N ≧ 2)
 * ***********************************************************************
 * 
 *   次にその中でも一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つ
 * の角にクイーンを置くことはできないので(効き筋だから）、ユニーク解であるかどうか
 * を判定するには、右上角から左下角を通る斜軸で反転させたパターンとの比較だけになり
 * ます。突き詰めれば、
 * 
 * [上から２行目のクイーンの位置が右から何番目にあるか]
 * [右から２列目のクイーンの位置が上から何番目にあるか]
 * 
 *
 * を比較するだけで判定することができます。この２つの値が同じになることはないからです。
 * 
 *       3 0
 *       ↓↓
 * - - - - Q ←0
 * - Q - - - ←3
 * - - - - -         上から２行目のクイーンの位置が右から４番目にある。
 * - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
 * - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
 * 
 *   結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
 * られる解は総てユニーク解であることが保証されます。
 * 
 * - - - - X Q
 * - Q - - X -
 * - - - - X -
 * - - - - X -
 * - - - - - -
 * - - - - - -
 * 
 *   次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
 * ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
 * ます。
 * 
 * X X - - - Q X X
 * X - - - - - - X
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * - - - - - - - -
 * X - - - - - - X
 * X X - - - - X X
 * 
 *   次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
 * 性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
 * て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
 * になります。
 * 
 * X X x x x Q X X
 * X - - - x x x X
 * C - - x - x - x
 * - - x - - x - -
 * - x - - - x - -
 * x - - - - x - A
 * X - - - - x - X
 * X X B - - x X X
 *
 *
 * ■ユニーク解から全解への展開
 *   これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
 * ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
 * 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
 * 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
 * を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
 * れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
 * いう考察が必要になってきます。
 * 
 *   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
 * オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
 * 角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
 * ず８個(＝２×４)になります。
 * 
 *   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
 * ために次の事柄を確認します。
 *
 * TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
 *   (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
 *    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
 *    ジナルと同型になる。  
 *
 *    COUNT2 * 2
 * 
 *   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
 *    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
 *    り得る。 
 *
 *    COUNT4 * 4
 * 
 *   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
 *       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
 *       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
 *       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 * 
 *    COUNT8 * 8 
 *
 *   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 * ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 * りを外す必要がなくなったというわけです。 
 * 
 *   UNIQUE  COUNT2      +  COUNT4      +  COUNT8
 *   TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)
 *
 * 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 * 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 * ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛
 * 躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
*/
void *run(void *args){
  struct local *l=(struct local *)args;
  int bit ;
  long COUNT2=l->COUNT2=0;
  long COUNT4=l->COUNT4=0;
  long COUNT8=l->COUNT8=0;
  long *c2=&(COUNT2);
  long *c4=&(COUNT4);
  long *c8=&(COUNT8);
  int SIZE=l->SIZE;
  int SIZEE =l->SIZEE;
  l->MASK=(1<<SIZE)-1;
  int BOUND1=l->BOUND1;
  int BOUND2=l->BOUND2;
  /* 最上段のクイーンが角にある場合の探索 */
  if(BOUND1>1 && BOUND1<SIZEE) { 
    l->aBoard[1]=bit=(1<<BOUND1);// 角にクイーンを配置 
    backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1),l,c8); 
  }//２行目から探索
  l->TOPBIT=1<<SIZEE;
  l->ENDBIT=(l->TOPBIT>>l->BOUND1);
  l->SIDEMASK=(l->TOPBIT|1);
  l->LASTMASK=(l->TOPBIT|1);
  /* 最上段行のクイーンが角以外にある場合の探索 */
  //   ユニーク解に対する左右対称解を予め削除するため片半分だけにクイーンを配置する
  if(BOUND1>0 && BOUND2<SIZEE && BOUND1<BOUND2){ 
    for(int i=1; i<BOUND1; i++){
      l->LASTMASK=l->LASTMASK|l->LASTMASK>>1|l->LASTMASK<<1; }
    l->aBoard[0]=bit=(1<<BOUND1);
    backTrack2(1,bit<<1,bit,bit>>1,
      l,
      c2,c4,c8); 
    l->ENDBIT>>=1; }
  l->COUNT2=*c2;
  l->COUNT4=*c4;
  l->COUNT8=*c8;
  return 0;
}
/**********************************************/
/*　マルチスレッド　*/
/**********************************************/
/**
 *
 * N=8の場合は8つのスレッドがおのおののrowを担当し処理を行います。

      メインスレッド  N=8
          +--BOUND1=7----- run()
          +--BOUND1=6----- run()
          +--BOUND1=5----- run()
          +--BOUND1=4----- run()
          +--BOUND1=3----- run()
          +--BOUND1=2----- run()
          +--BOUND1=1----- run()
          +--BOUND1=0----- run()
  
 * そこで、それぞれのスレッド毎にスレッドローカルな構造体を持ちます。
 *
      // スレッドローカル構造体 
      struct local{
        int bit;
        int BOUND1;
        int BOUND2;
        int TOPBIT;
        int ENDBIT;
        int MASK;
        int SIDEMASK;
        int LASTMASK;
        int aBoard[MAXSIZE];
      };
 * 
 * スレッドローカルな構造体の宣言は以下の通りです。
 *
 *    //スレッドローカルな構造体
 *    struct local l[MAXSIZE];
 *
 * アクセスはグローバル構造体同様 . ドットでアクセスします。
      l[BOUND1].BOUND1=BOUND1;
      l[BOUND1].BOUND2=BOUND2;
 *
 */
/**********************************************/
/* マルチスレッド　排他処理  mutex            */
/**********************************************/
/**
 * マルチスレッド pthreadには排他処理 mutexがあります。
   まずmutexの宣言は以下の通りです。

  // mutexの宣言
  pthread_mutex_t mutex;   
  //pthread_mutexattr_t 変数を用意します。
  pthread_mutexattr_t mutexattr;
  // pthread_mutexattr_t 変数にロック方式を設定します。
  pthread_mutexattr_init(&mutexattr);
  //以下の第二パラメータでロック方式を指定できます。（これはとても重要です）
    PTHREAD_MUTEX_NORMAL  PTHREAD_MUTEX_FAST_NP 
    誰かがロックしているときに、それが解放されるまで永遠に待ちます。
    （同一スレッド内でのロックもブロック、その代り動作が速い）

    PTHREAD_MUTEX_RECURSIVE PTHREAD_MUTEX_RECURSIVE_NP  
    誰かがロックしているときに、それが解放されるまで永遠に待ちます。
    （同一スレッド内での２度目以降のロックは素通り）

    PTHREAD_MUTEX_ERRORCHECK  PTHREAD_MUTEX_ERRORCHECK_NP 
    誰かがロックしているときに、直ちに EDEADLK (11) を戻り値に返します。
    （同一スレッド内で 2 度目のロックがあったことを検出できる）      

    <>第 2 引数で NULL を指定した場合は、PTHREAD_MUTEX_NORMAL が指定されたのと同じになります。

  pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_RECURSIVE);
  // ミューテックスを初期化します。
  pthread_mutex_init(&mutex, &mutexattr);
  //pthread_mutex_init(&mutex, NULL); // 通常はこう書きますが遅いです

  実際にロックする場合はできるだけ局所的に以下の構文を挟み込むようにします。
  //pthread_mutex_lock(&mutex);
  //pthread_mutex_unlock(&mutex);
 
 * 実行部分は以下のようにロックとロック解除で処理を挟みます。
      pthread_mutex_lock(&mutex);     //ロックの開始
        COUNT2+=C2;                //保護されている処理
        COUNT4+=C4;                //保護されている処理
        COUNT8+=C8;                //保護されている処理
      pthread_mutex_unlock(&mutex);   //ロックの終了
 *
  使い終わったら破棄します。
    pthread_mutexattr_destroy(&mutexattr);//不要になった変数の破棄
    pthread_mutex_destroy(&mutex);        //nutexの破棄
 *
 */
/**
  ですが、mutexのロックとロック解除は処理の中断と開始が頻繁に走り、
　速度が著しく低下します。
  そこで、スレッド毎に独立した配列にそれぞれに変数を格納し、
　スレッドセーフなアトミック対応を行います。

   mutex１つをロック・ロック解除で使い回すことでボトルネックが発生しました。
   また、mutexをスレッドの数だけ生成し、スレッド毎にロック/ロック解除を
   繰り返すことでオーバーヘッドは少なくなったものの、依然としてシングルスレッ
   ドよりも速度は遅くなることとなりました。

   高速化を実現するならばmutexで排他処理を行うよりも、アトミックに
   メモリアクセスする方が良さそうです。
   排他処理に必要な箇所はCOUNT++する箇所となります。
   具体的にはカウントする変数をスレッド毎の配列に格納し、
   COUNT2[BOUND1] COUNT4[BOUND1] COUNT8[BOUND1]で実装します。

  // mutexを廃止したことで以下の宣言が不要となりました。
   //pthread_mutexattr_t 変数を用意します。
   pthread_mutexattr_t mutexattr;
   //pthread_mutexattr_t 変数にロック方式を設定します。
   pthread_mutexattr_init(&mutexattr);
   pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_NORMAL);
*/
void *NQueenThread( void *args){
  struct local l[MAXSIZE]; //構造体 local型 
  int SIZE=*(int *)args; int SIZEE=SIZE-1; //argsで引き渡されたパラメータをSIZEに格納
  pthread_t cth[SIZE]; //スレッド childThread
  // B1から順にスレッドを生成しながら処理を分担する 
  for(int B1=SIZEE,B2=0;B2<SIZEE;B1--,B2++){ 
    l[B1].BOUND1=B1; l[B1].BOUND2=B2; l[B1].SIZE=SIZE; l[B1].SIZEE=SIZEE; //スレッド毎の変数の初期化
    for(int j=0;j<SIZE;j++){ l[B1].aBoard[j]=j; } 
    pthread_create(&cth[B1],NULL,run, (void *) &l[B1]); // マルチスレッドの生成 
  }
  for(int B1=SIZEE,B2=0;B2<SIZEE;B1--,B2++){ pthread_join(cth[B1],NULL); } //処理が終わったら 全てjoin
  for(int B1=SIZEE,B2=0;B2<SIZEE;B1--,B2++){ //スレッド毎のカウンターを合計
    G.lTotal+=l[B1].COUNT2*2+l[B1].COUNT4*4+l[B1].COUNT8*8;
    G.lUnique+=l[B1].COUNT2+l[B1].COUNT4+l[B1].COUNT8; 
  }
  return 0;
}
/**********************************************/
/*  マルチスレッド pthread                    */
/**********************************************/
/**
 *  マルチスレッドには pthreadを使います。
 *  pthread を宣言するには pthread_t 型の変数を宣言します。
 *
      pthread_t pth;  //スレッド変数
      
      マルチスレッドの生成
      pthread_create(&pth, NULL, スレッドしたい関数,渡したい構造体または変数); 
    
    pthread_create()は渡したい関数に１つしか変数を渡せません。
    ですから、複数の変数を渡したい場合は、構造体にまとめて渡します。

  //構造体
  struct local{
    int BOUND1;
    int BOUND2;
    int aBoard[MAXSIZE];
    int SIZE;
    int SIZEE;
    long COUNT2;
    long COUNT4;
    long COUNT8;
  };

  渡された関数側
  void *run(void *args){
    struct local *l=(struct local *)args;
    int SIZE=l->SIZEE; //こんな感じで
  }  

  スレッドを生成するには pthread_create()を呼び出します。
  戻り値iFbRetにはスレッドの状態が格納されます。正常作成は0になります。
  pthread_join()はスレッドの終了を待ちます。
 */
void NQueen(int SIZE){
  pthread_t pth;  //スレッド変数
  pthread_create(&pth, NULL, NQueenThread,(void *) &SIZE); // メインスレッドの生成
  pthread_join(pth, NULL); //スレッドの終了を待つ
}
/**********************************************/
/*  メイン関数                                */
/**********************************************/
/**
 * N=2 から順を追って 実行関数 NQueen()を呼び出します。
 * 最大値は 先頭行でMAXSIZEをdefineしています。
 * G は グローバル構造体で宣言しています。

    //グローバル構造体
    typedef struct {
      int nThread;
      int SIZE;
      int SIZEE;
      long COUNT2;
      long COUNT4;
      long COUNT8;
    }GCLASS, *GClass;
    GCLASS G; //グローバル構造体

グローバル構造体を使う場合は
  G.SIZE=i ; 
  のようにドットを使ってアクセスします。
 
  NQueen()実行関数は forの中の値iがインクリメントする度に
  Nのサイズが大きくなりクイーンの数を解法します。 
 */
int main(void){
  printf("%s\n"," N:           Total           Unique          hh:mm:ss.ms");
  struct timeval t0; struct timeval t1;
  for(int i=2;i<=MAXSIZE;i++){
    G.lTotal=G.lUnique=0;
    gettimeofday(&t0, NULL);NQueen(i);gettimeofday(&t1, NULL);
    int ss;int ms;int dd;
    if (t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400; 
      ss=(t1.tv_sec-t0.tv_sec-1)%86400; 
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000; 
    } else { 
      dd=(t1.tv_sec-t0.tv_sec)/86400; 
      ss=(t1.tv_sec-t0.tv_sec)%86400; 
      ms=(t1.tv_usec-t0.tv_usec+500)/10000; 
    }
    int hh=ss/3600; 
    int mm=(ss-hh*3600)/60; 
    ss%=60;
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,G.lTotal,G.lUnique,dd,hh,mm,ss,ms); 
  } 
}
