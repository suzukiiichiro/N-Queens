/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 実行
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC03.c && ./a.out [-c|-r]


 ３．バックトラック

 　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
 さない事が明らかな場合は、それ以降のパターン生成を行わない。
 　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
 　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
 は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
 けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
 を降りても解がないと判明した時点で木を引き返すということができます。


  N-Queen の データ配列について
  =============================

  総当たり
  結局全部のケースをやってみる（完全解）

  バックトラック
  とりあえずやってみる。ダメなら戻って別の道を探る


  N-Queen: クイーンの効き筋
  =========================
  クイーンの位置から、縦、横、斜めが効き筋となります。

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|-*-|---|---|-*-|---|
        +-------------------+
       1|---|-*-|---|-*-|---|
        +-------------------+ 
       2|---|---|-*-|-*-|-*-| 
        +-------------------+ 
       3|-*-|-*-|-*-|-Q-|-*-|
        +-------------------+
       4|---|---|-*-|-*-|-*-|
        +-------------------+


  N-Queen: 盤面上で互いのクイーンが効き筋にならないように配置
  ===========================================================

        完成図は以下の通りです。

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---|
        +-------------------+
       1|---|---|---|-Q-|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|---|---|---|---|-Q-|
        +-------------------+
       4|---|---|-Q-|---|---|
        +-------------------+


  効き筋の表現
  ============

  クイーンの位置から下側を走査対象とします。

  　すでに効き筋：FALSE(盤面ではF）
  　配置可能    ：TRUE

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---| 
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|-F-|-F-|-F-|---|---|
        +-------------------+
       4|---|-F-|---|-F-|---|
        +-------------------+
                      

  効き筋を三つの配列で表現
  ========================

  ■ 基本：aBoard[row]=col
           aBoard[2  ]=1

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| aBoard[2]=1 に配置
        +-------------------+
       3|---|---|---|---|---|
        +-------------------+
       4|---|---|---|---|---|
        +-------------------+


  ■配列1：down[row]

  そのrow(行)にQueenがいる場合はFALSE
                      いない場合はTRUE

  　　       column(列)
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+
       3|---|-F-|---|---|---|
        +-------------------+
       4|---|-F-|---|---|---|
        +-------------------+
             down[col(1)]==false (すでに効き筋）


  ■配列２：right[col-row+N-1]
                    right[col-row+N-1]==F
                        Qの場所：col(1)-row(2)+(4-1)=2なので
                        col-row+N-1が２のところがＦとなる 
  　　       column(列)
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|---|---|-F-|---|---|
        +-------------------+
       4|---|---|---|-F-|---|
        +-------------------+
                      right[col-row+(N-1)]==false(すでに効き筋）


  ■配列3：left[col+row]
                      left[col+row]==F 
                          Qの場所：col(1)+row(2)=3なので
                          col+rowが3になるところがFとなる。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|---|---|---|---|---|
        +-------------------+
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|-Q-|---|---|---| 
        +-------------------+ 
       3|-F-|---|---|---|---|
        +-------------------+
       4|---|---|---|---|---|
        +-------------------+
      left[col+row]


  ステップ１
  ==========
  row=0, col=0 にクイーンを配置してみます。

  aBoard[row]=col
     ↓
  aBoard[0]=0;

  　　       column(列) 
  row(行)_0___1___2___3___4_
   ->  0|-Q-|---|---|---|---| aBoard[row]=col
        +-------------------+ aBoard[0  ]=0  
       1|---|---|---|---|---|
        +-------------------+ 
       2|---|---|---|---|---| 
        +-------------------+ 
       3|---|---|---|---|---|
        +-------------------+
       4|---|---|---|---|---|
        +-------------------+


  考え方：２
  ==========
  効き筋を埋めます

  　　       column(列) 
  row(行)_0___1___2___3___4_
   ->  0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|---|---|---|
        +-------------------+ left はありません
       2|-F-|---|-F-|---|---| 
        +-------------------+ 
       3|-F-|---|---|-F-|---|
        +-------------------+
       4|-F-|---|---|---|-F-|
        +-------------------+
        down[col]      right[col-row+(N-1)]


  考え方：３
  ==========
  rowが一つ下に降りて０から１となります。
  次の候補は以下のＡ，Ｂ，Ｃとなります

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
   ->  1|-F-|-F-|-A-|-B-|-C-|
        +-------------------+ 
       2|-F-|---|-F-|---|---| 
        +-------------------+ 
       3|-F-|---|---|-F-|---|
        +-------------------+
       4|-F-|---|---|---|-F-|
        +-------------------+

  考え方：４
  ==========
  Ａにおいてみます。
  効き筋は以下の通りです。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
   ->  1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|---| 
        +-------------------+ 
       3|-F-|---|-F-|-F-|-F-| right[col-row+(N-q)]
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+
  left[col+row]  down[col]


  考え方：５
  ==========
  rowが一つ下に降りて１から２となります。
  次の候補はＡとなります

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
   ->  2|-F-|-F-|-F-|-F-|-A-| 
        +-------------------+ 
       3|-F-|---|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+

  考え方：６
  ==========
  効き筋は以下の通りです。
  特に加わるところはありません。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
   ->  2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
       3|-F-|---|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+

  考え方：７
  ==========
  rowが一つ下に降りて２から３となります。
  次の候補はＡとなります

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
   ->  3|-F-|-A-|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|---|-F-|---|-F-|
        +-------------------+


  考え方：８
  ==========
  効き筋は以下の通りです。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
   ->  3|-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
       4|-F-|-F-|-F-|---|-F-|
        +-------------------+


  考え方：９
  ==========
  今回は、うまくいっていますが、
  次の候補がなければ、キャンセルして、
  前のコマを次の候補にコマを移動し、
  処理を継続します。


  考え方：１０
  =========-=

  rowが一つ下に降りて３から４となります。
  候補はのこり１箇所しかありません。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
       3|-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
   ->  4|-F-|-F-|-F-|-A-|-F-|
        +-------------------+



  考え方：１１
  ==========
  最後のクイーンをおきます
  columnの最終列は効き筋を確認する必要はありませんね。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---| 
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-| 
        +-------------------+ 
       3|-F-|-Q-|-F-|-F-|-F-| 
        +-------------------+
   ->  4|-F-|-F-|-F-|-Q-|-F-|
        +-------------------+

  考え方：１２
  ==========
  rowの脇にcolの位置を示します。

  　　       column(列) 
  row(行)_0___1___2___3___4_
       0|-Q-|---|---|---|---|  [0]
        +-------------------+ 
       1|-F-|-F-|-Q-|---|---|  [2]
        +-------------------+ 
       2|-F-|-F-|-F-|-F-|-Q-|  [4]
        +-------------------+ 
       3|-F-|-Q-|-F-|-F-|-F-|  [1]
        +-------------------+
   ->  4|-F-|-F-|-F-|-Q-|-F-|  [3]
        +-------------------+


  考え方：１３
  ==========

  ボード配列は以下のように表します。
  aBoard[]={0,2,4,1,3]

  出力：
    1: 0 0 0 1
    2: 0 0 0 2
    3: 0 0 0 4
    :


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC03.c && ./a.out -r
３．CPUR 再帰 バックトラック
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.27
14:       365596               0            1.66
15:      2279184               0           10.63
16:     14772512               0         1:12.95
17:     95815104               0         8:40.60


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC03.c && ./a.out -c
３．CPU 非再帰 バックトラック
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.01
12:        14200               0            0.05
13:        73712               0            0.29
14:       365596               0            1.73
15:      2279184               0           11.38
16:     14772512               0         1:14.90
17:     95815104               0         9:00.09
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//変数宣言
int aBoard[MAX];
int down[2*MAX-1];   //down:flagA 縦 配置フラグ
int right[2*MAX-1];  //right:flagB 斜め配置フラグ
int left[2*MAX-1]; //left:flagC 斜め配置フラグ
long TOTAL=0;
long UNIQUE=0;
//関数宣言
void TimeFormat(clock_t utime,char *form);
void NQueen(int row,int size);
void NQueenR(int row,int size);
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char* form){
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
//CPU 非再帰版 ロジックメソッド
void NQueen(int row,int size){
  int sizeE=size-1;
  bool matched;
  while(row>=0){
    matched=false;
    // １回目はaBoard[row]が-1なのでcolを0で初期化
    // ２回目以降はcolを<sizeまで右へシフト
    for(int col=aBoard[row]+1;col<size;col++){
      if(down[col]==0
          &&right[col-row+sizeE]==0
          &&left[col+row]==0){   //まだ効き筋がない
        if(aBoard[row]!=-1){    //Qを配置済み
          //colがaBoard[row]におきかわる
          down[aBoard[row]]
            =right[aBoard[row]-row+sizeE]
            =left[aBoard[row]+row]=0;
        }
        aBoard[row]=col;        //Qを配置
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=1;      //効き筋とする
        matched=true;            //配置した
        break;
      }
    }
    if(matched){                //配置済みなら
      row++;                    //次のrowへ
      if(row==size){
        //print(size); //print()でTOTALを++しない
        TOTAL++;
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        int col=aBoard[row]; /** col の代用 */
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=0;
        aBoard[row]=-1;
      }
      row--;                    //バックトラック
    }
  }
}
// CPUR 再帰版 ロジックメソッド
void NQueenR(int row,int size){
  int sizeE=size-1;
  if(row==size){
    TOTAL++;
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;    //Qを配置
      if(down[col]==0
          && right[row-col+sizeE]==0
          && left[row+col]==0){
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=1;
        NQueenR(row+1,size);
        down[col]
          =right[row-col+sizeE]
          =left[row+col]=0;
      }
      aBoard[row]=-1;     //空き地に戻す
    }
  }
}
//メインメソッド
int main(int argc,char** argv){
  bool cpu=false,cpur=false;
  int argstart=2;
  /** 起動パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else{ cpur=true;}
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
  if(cpu){
    printf("\n\n３．CPU 非再帰 バックトラック\n");
  }else if(cpur){
    printf("\n\n３．CPUR 再帰 バックトラック\n");
  }
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  clock_t st;           //速度計測用
  char t[20];           //hh:mm:ss.msを格納
  int min=4;
  int targetN=17;
  for(int i=min;i<=targetN;i++){
    TOTAL=0;
    UNIQUE=0;
    st=clock();
    // aBoard配列の初期化
    for(int j=0;j<=targetN;j++){ aBoard[j]=-1; }
    /**  非再帰 */
    if(cpu){ NQueen(0,i); }
    /**  再帰 */
    if(cpur){ NQueenR(0,i); }
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
  }
  return 0;
}
