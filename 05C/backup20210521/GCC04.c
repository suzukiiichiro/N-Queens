/**
  Cで学ぶアルゴリズムとデータ構造
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)


 実行
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC04.c && ./a.out [-c|-r]


 ４．バックトラック＋対称解除法

 　一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
 　８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 
 ■ユニーク解の判定方法
   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 ことを意味しています。
 
   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 わりません。
 
   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 下の例では要素数が２個のものと８個のものがあります。


 Ｎ＝５の全解は１０、ユニーク解は２なのです。
 
 グループ１: ユニーク解１つ目
 - - - Q -   - Q - - -
 Q - - - -   - - - - Q
 - - Q - -   - - Q - -
 - - - - Q   Q - - - -
 - Q - - -   - - - Q -
 
 グループ２: ユニーク解２つ目
 - - - - Q   Q - - - -   - - Q - -   - - Q - -   - - - Q -   - Q - - -   Q - - - -   - - - - Q
 - - Q - -   - - Q - -   Q - - - -   - - - - Q   - Q - - -   - - - Q -   - - - Q -   - Q - - -
 Q - - - -   - - - - Q   - - - Q -   - Q - - -   - - - - Q   Q - - - -   - Q - - -   - - - Q -
 - - - Q -   - Q - - -   - Q - - -   - - - Q -   - - Q - -   - - Q - -   - - - - Q   Q - - - -
 - Q - - -   - - - Q -   - - - - Q   Q - - - -   Q - - - -   - - - - Q   - - Q - -   - - Q - -

 
   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 と呼ぶことにします。
 
 - - - - Q   0
 - - Q - -   2
 Q - - - -   4   --->  0 2 4 1 3  (ユニーク判定値)
 - - - Q -   1
 - Q - - -   3
 
 
   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 変換を試みるだけでユニーク解の判定が可能になります。
  
 
 ■ユニーク解の個数を求める
   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 央になければならず、それは互いの効き筋にあたるので有り得ません。

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
  - - - - Q ←0
  - Q - - - ←3
  - - - - -         上から２行目のクイーンの位置が右から４番目にある。
  - - - Q -         右から２列目のクイーンの位置が上から４番目にある。
  - - - - -         しかし、互いの効き筋にあたるのでこれは有り得ない。
  
    結局、再帰探索中において下図の X への配置を禁止する枝刈りを入れておけば、得
  られる解は総てユニーク解であることが保証されます。
  
  - - - - X Q
  - Q - - X -
  - - - - X -
  - - - - X -
  - - - - - -
  - - - - - -
  
    次に右端以外にクイーンがある場合を考えてみます。オリジナルがユニーク解である
  ためには先ず下図の X への配置は禁止されます。よって、その枝刈りを先ず入れておき
  ます。
  
  X X - - - Q X X
  X - - - - - - X
  - - - - - - - -
  - - - - - - - -
  - - - - - - - -
  - - - - - - - -
  X - - - - - - X
  X X - - - - X X
  
    次にクイーンの利き筋を辿っていくと、結局、オリジナルがユニーク解ではない可能
  性があるのは、下図の A,B,C の位置のどこかにクイーンがある場合に限られます。従っ
  て、90度回転、180度回転、270度回転の３通りの変換パターンだけを調べれはよいこと
  になります。
  
  X X x x x Q X X
  X - - - x x x X
  C - - x - x - x
  - - x - - x - -
  - x - - - x - -
  x - - - - x - A
  X - - - - x - X
  X X B - - x X X
 
 
  ■ユニーク解から全解への展開
    これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
  ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
  探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
  導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
  を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
  れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
  いう考察が必要になってきます。
  
    最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
  オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
  角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
  ず８個(＝２×４)になります。
  
    次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
  ために次の事柄を確認します。
 
  TOTAL = (COUNT8 * 8) + (COUNT4 * 4) + (COUNT2 * 2);
    (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
     ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
     ジナルと同型になる。  
 
     COUNT2 * 2
  
    (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
     ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
     り得る。 
 
     COUNT4 * 4
  
    (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
        加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
        180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
        オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
  
     COUNT8 * 8 
 
    以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
  ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
  りを外す必要がなくなったというわけです。 
  
    UNIQUE  COUNT2      +  COUNT4      +  COUNT8
    TOTAL  (COUNT2 * 2) + (COUNT4 * 4) + (COUNT8 * 8)
 
  　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
  　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
  ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛
  躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC04.c && ./a.out -r
４．CPUR 再帰 バックトラック＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1787            0.05
13:        73712            9233            0.28
14:       365596           45752            1.65
15:      2279184          285053           10.61
16:     14772512         1846955         1:12.29
17:     95815104        11977939         8:42.16


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC04.c && ./a.out -c
４．CPU 非再帰 バックトラック＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.01
12:        14200            1787            0.05
13:        73712            9233            0.28
14:       365596           45752            1.71
15:      2279184          285053           10.92
16:     14772512         1846955         1:13.64
17:     95815104        11977939         8:46.42
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//
int aBoard[MAX];
int down[2*MAX-1];   //down:flagA 縦 配置フラグ
int right[2*MAX-1]; //right:flagB 斜め配置フラグ
int left[2*MAX-1];   //left:flagC 斜め配置フラグ
long TOTAL=0;
long UNIQUE=0;
int aT[MAX];         //aT:aTrial[]
int aS[MAX];         //aS:aScrath[]
//関数宣言
void TimeFormat(clock_t utime,char *form);
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);
int symmetryOps(int size);
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
//回転
void rotate(int chk[],int scr[],int n,int neg){
  int k=neg ? 0 : n-1;
  int incr=(neg ? +1 : -1);
  for(int j=0;j<n;k+=incr){
    scr[j++]=chk[k];
  }
  k=neg ? n-1 : 0;
  for(int j=0;j<n;k-=incr){
    chk[scr[j++]]=k;
  }
}
//反転
void vMirror(int chk[],int n){
  for(int j=0;j<n;j++){
    chk[j]=(n-1)-chk[j];
  }
}
//
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){
      break;
    }
  }
  return rtn;
}
//対称解除法
int symmetryOps(int size){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aT[i]=aBoard[i];
  }
  //時計回りに90度回転
  rotate(aT,aS,size,0);
  int k=intncmp(aBoard,aT,size);
  if(k>0) return 0;
  if(k==0){
    nEquiv=1;
  }else{
    //時計回りに180度回転
    rotate(aT,aS,size,0);
    k=intncmp(aBoard,aT,size);
    if(k>0) return 0;
    if(k==0){
      nEquiv=2;
    }else{
      //時計回りに270度回転
      rotate(aT,aS,size,0);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<size;i++){
    aT[i]=aBoard[i];
  }
  //垂直反転
  vMirror(aT,size);
  k=intncmp(aBoard,aT,size);
  if(k>0){
    return 0;
  }
  //-90度回転 対角鏡と同等
  if(nEquiv>1){
    rotate(aT,aS,size,1);
    k=intncmp(aBoard,aT,size);
    if(k>0){
      return 0;
    }
    //-180度回転 水平鏡像と同等
    if(nEquiv>2){
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
      //-270度回転 反対角鏡と同等
      rotate(aT,aS,size,1);
      k=intncmp(aBoard,aT,size);
      if(k>0){
        return 0;
      }
    }
  }
  return nEquiv*2;
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
          && right[col-row+sizeE]==0
          && left[col+row]==0){ //まだ効き筋がない
        if(aBoard[row]!=-1){    //Qを配置済み
          //colがaBoard[row]におきかわる
          down[aBoard[row]]
            =right[aBoard[row]-row+sizeE]
            =left[aBoard[row]+row]=0;
        }
        aBoard[row]=col;        //Qを配置
        down[col]
          =right[col-row+sizeE]
          =left[col+row]=1;     //効き筋とする
        matched=true;           //配置した
        break;
      }
    }
    if(matched){                //配置済みなら
      row++;                    //次のrowへ
      if(row==size){
        //print(size); //print()でTOTALを++しない
        /** 対称解除法の導入 */
        int s=symmetryOps(size);
        if(s!=0){
          UNIQUE++;             //ユニーク解を加算
          TOTAL+=s;   //対称解除で得られた解数を加算
        }
        // TOTAL++;
        /** 対称解除法の導入 */
        row--;
      }
    }else{
      if(aBoard[row]!=-1){
        int col=aBoard[row];    /** col の代用 */
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
    /** 対称解除法の導入 */
    int s=symmetryOps(size);
    if(s!=0){
      UNIQUE++;       //ユニーク解を加算
      TOTAL+=s;       //対称解除で得られた解数を加算
    }
    // TOTAL++;
    /** 対称解除法の導入 */
  }else{
    for(int col=aBoard[row]+1;col<size;col++){
      aBoard[row]=col;
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
      aBoard[row]=-1;
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
    printf("\n\n４．CPU 非再帰 バックトラック＋対称解除法\n");
  }else if(cpur){
    printf("\n\n４．CPUR 再帰 バックトラック＋対称解除法\n");
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
