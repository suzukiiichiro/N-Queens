/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 実行
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC01.c && ./a.out [-c|-r]


 1. ブルートフォース　力任せ探索

 　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解とな
 るかをチェックする方法

   (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
   けであって8王妃問題を解いているわけではありません


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
    :



 実行結果
 1. CPUR 再帰 ブルートフォース　力任せ探索
 :
 :
3118: 4 4 4 3 2
3119: 4 4 4 3 3
3120: 4 4 4 3 4
3121: 4 4 4 4 0
3122: 4 4 4 4 1
3123: 4 4 4 4 2
3124: 4 4 4 4 3
3125: 4 4 4 4 4

 1. CPU 非再帰 ブルートフォース　力任せ探索
 :
 :
3118: 4 4 4 3 2
3119: 4 4 4 3 3
3120: 4 4 4 3 4
3121: 4 4 4 4 0
3122: 4 4 4 4 1
3123: 4 4 4 4 2
3124: 4 4 4 4 3
3125: 4 4 4 4 4

 **/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
//
#define MAX 8
//変数宣言
int aBoard[MAX]; //版の配列
int COUNT=0;     //カウント用
//関数宣言
void print(int size);
void NQueen(int row,int size);
void NQueenR(int row,int size);
//出力
void print(int size){
  printf("%d: ",++COUNT);
  for(int j=0;j<size;j++){
    printf("%d ",aBoard[j]);
  }
  printf("\n");
}
//CPU 非再帰版ロジックメソッド
void NQueen(int row,int size){
  bool matched;
  while(row>=0){
    matched=false;
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;      //Qを配置
      matched=true;
      break;
    }
    if(matched){
      row++;
      if(row==size){
        print(size);
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        aBoard[row]=-1;
      }
      row--;
    }
  }
}
//CPUR 再帰版ロジックメソッド
void NQueenR(int row,int size){
  if(row==size){         //SIZEは5で固定
    print(size);         //rowが5になったら出力
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;  //Qを配置
      NQueenR(row+1,size);
      aBoard[row]=-1;   //空き地に戻す
    }
  }
}
//メインメソッド
int main(int argc,char** argv){
  int size=5;
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
  // aBoard配列の初期化
  for(int i=0;i<size;i++){ aBoard[i]=-1; }
  /**  非再帰 */
  if(cpu){ NQueen(0,size); }
  /**  再帰 */
  if(cpur){ NQueenR(0,size); }
  return 0;
}
