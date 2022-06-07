/**

 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

９−３．CPUR 再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2
 N:        Total       Unique        hh:mm:ss.ms
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1788            0.01
13:        73712            9237            0.04
14:       365596           45771            0.23
15:      2279184          285095            1.40

BackTrack1,2のいずれの枝刈りも回転・ミラーして点数が低いものは枝刈りして落としてしまおうというもの。
symmetryOpsで点数が低いものは落とされるので
枝刈りをするならsymmetryOpsより前に置くべきで、
枝刈りの内容を見てみるといずれも2行2列にクイーンを置く際に判定できるものなので
board_placementsに置くべきと考え設置してみた。


[BackTrack2]
1行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを置くことはできない
置いた場合、回転・鏡像変換により得られる状態のユニーク判定値が明らかに大きくなる

☓☓・・・Ｑ☓☓
☓・・・／｜＼☓
ｃ・・／・｜・rt
・・／・・｜・・
・／・・・｜・・
lt・・・・｜・ａ
☓・・・・｜・☓
☓☓ｂ・・dn☓☓

->ユニーク判定値の関係でxx にクイーンを置くことはできない
Qと同じ行の右2つが「クイーン位置より右位置」これを90度回転、ミラーするとxの部分になる。
BackTrack2の枝刈りはこのxの部分にクイーンを置けないというもの
Qと同じ行にあるxはクイーンの効き筋なので枝刈りの必要はない
枝刈り対象は以下の3パターンになる

1 上部サイド枝刈り
2 下部サイド枝刈り
3 最下段枝刈り

1x---qx1
1------1
--------
--------
--------
--------
2------2
23----32


  
【枝刈り】上部サイド枝刈り  
 row<BOUND1 の時は両端にクイーンを置けないというもの
 ->左端1行、右端1行にクイーンを置く際に判定できる
  
【枝刈り】下部サイド枝刈り
row>BOUND2 の時は両端にクイーンを置けないというもの
 ->左端1行、右端1行にクイーンを置く際に判定できる

【枝刈り】最下段枝刈り
最終行でLASTMASKにひっかかるものはクイーンを置けないというもの
例えばBOUND1(1行めのクイーンの位置)が2の時は最終行の左端2列、右端2列にクイーンを置けない
->最終行にクイーンを置く際に判定できる

[backTrack1]
　1、1行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
　　  1行目角にクイーンがある場合、他の角にクイーンを配置することは不可
    ->90度回転、180度回転させて同じになることはないので正解がある場合は必ず8個
    ->symmetryOpsで2,4,8を判定させる必要はない
    　
  2、【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      2行目、2列目を数値とみなし、2行目＜2列目という条件を課せばよい
    ->主体角線鏡像とは、１行目の角を左上から最終行の右下角の線を軸に反転させるということ
    ->主体角線鏡像で反転させると２行目のクイーンと２列目のクイーンが入れ替わる
    ->ミラーだと得点の低い方が採用される
    ->例えば、2行目のクイーンの位置が4列目、2列目のクイーンの位置が3行目だとすると、主体角線鏡像でミラー反転させると、２行目のクイーンの位置が3列目、2列目のクイーンの位置が4行目となる。
    ->小さい行数の列数がより小さい方が得点が小さい
    ->今回の例だとミラー反転前は 1行目 0  2行目 4 ミラー反転後は 1行目 0 2行目 3 になるのでミラー反転させた方が得点が小さくなるので不採用になる（枝刈りして良い）

    ->例えば、2行目のクイーンの位置が4列目、2列目のクイーンの位置が5行目だとすると、主体角線鏡像でミラー反転させると、2行目のクイーンの位置が5列目、2列目のクイーンの位置が4行目となる。
    ->今回の例だとミラー反転前は 1行目 0  2行目 4 ミラー反転後は 1行目 0 2行目 5 になるのでミラー反転させた方が得点が大きくなるので採用になる（枝刈り不要）

    ->2行目の列数 ＜ 2列目の行数 になるようにすれば良い
    ->BOUND1<row の時は2列目にクイーンを置けないようにする
    -> bitmap&=~2; 
    ->例えば bitamap が 1000010 だと bitmap&=~2で 1000000 になる


  pressに枝刈りを追加する
  (1)
  symmetryOpsの関係で角にクイーンを置けるのは右上1箇所だけ
  それ以外に角に置こうとする場合はスキップして良い

  角に置けるのはここだけ

  ----o
  -----
  -----
  -----
  -----

  以下は置けない

  o----   -----   -----
  -----   -----   -----
  -----   -----   -----
  -----   -----   -----
  -----   o----   ----o

  (2)
  例えばN5の場合のpress_a,press_bの組み合わせは以下の通り
  (0,2),(0,3),(0,4),
  (1,3),(1,4),
  (2,0),(2,4),
  (3,0),(3,1),
  (4,0),(4,1),(4,2)
  
  上1行で右端にクイーンが置かれた場合は
  ----o
  -----
  -----
  -----
  -----
  
  右にクイーンを置くときには
  右1列目は固定で右2列目にどこを置くかだけ検討すれば良い
  presで見ると
  (0,2),(0,3),(0,4)だけ検討すれば良い
  (以下の図でxの部分だけ検討する)
  presの数は12->3個に減少する

  ----o
  -----
  ---x-
  ---x-
  ---x-

  (3)
  角から2番目にクイーンを置いたら、次の2行目は0固定
  例:上1行目で左から2列目にクイーンを置いたら、左2行目は0固定
  
  presで見ると
  (2,0)(3,0)(4,0)だけ検討すれば良い
  presの数は12->3個に減少する

  -o---
  -----
  x----
  x----
  x----

  バグがあると思われるところ
  while(bitmap>0){ が118回目くらい回ったところから同じ場所にクイーンを置く動きがある
    gdbでみたいところまでスキップする
    ・次のブレークポイントまで実行
c
https://qiita.com/arene-calix/items/a08363db88f21c81d351

・指定したループのところまでスキップしてみたいところから確認する方法が今回役に立ちそうです。

https://www.cism.ucl.ac.be/Services/Formations/ICS/ics_2013.0.028/composer_xe_2013/Documentation/ja_JP/debugger/cl/GUID-151D5058-C356-4931-A5F3-08B6FDBC950D.htm

ブレークポイントを貼った後gdbコマンドで入力
ignore ID count  

ignore 1 10 だと10回ブレークポイントを通ったところから確認できる

  C9:038----17
  000000001
  000001000
  100000000
  000000000
  000000000
  000000000
  000000000
  000000010
  010000000
  
  gcnt:118
  クイーンを配置
  N9:0384---17
  000000001
  000001000
  100000000
  000010000
  000000000
  000000000
  000000000
  000000010
  010000000
  
  gcnt:119
  クイーンを配置
  N9:03842--17
  000000001
  000001000
  100000000
  000010000
  000000100
  000000000
  000000000
  000000010
  010000000
  
  gcnt:120
  クイーンを配置
  N9:03862--17
  000000001
  000001000
  100000000
  001000000
  000000100
  000000000
  000000000
  000000010
  010000000
  
  gcnt:121
  クイーンを配置
  N9:03862--17
  000000001
  000001000
  100000000
  001000000
  000000100
  000000000
  000000000
  000000010
  010000000
  

*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
/**
 *
 */
long TOTAL=0;
long UNIQUE=0;
/**
 * ecntはステップ数を確認するためのデバッグ用の変数。
 */
long ecnt=0;
/**
 *
 *
 */
struct Board
{
  int row;
  int sym;
  long down;
  long left;
  long right;
  int aBoard[MAX];
  int bBoard[MAX];
  int topSide;
  int leftSide;
  int bottomSide;
  int rightSide;
  long COUNT[3];
};
/**
 * デバッグ用
 * 1行分のクイーンの位置を0,1で返却する
 * 256->010000000(N9の場合)
 */
void dec2bin(int size,int dec)
{
  int i,b[32];
  for (i=0;i<size;i++){
    b[i]=dec%2;
    dec=dec/2;
  }
  char* buf;
  while (i){ 
    printf("%2d ", b[--i]); 
  }
}
/**
 * デバッグ用
 * クイーンの配置を表示する
 *
 * N9:316807425<--各行のクイーンの列番号
 * 000001000   <--クイーンの置かれている場所が1、それ以外は0
 * 000000010
 * 001000000
 * 100000000
 * 000000001
 * 010000000
 * 000010000
 * 000000100
 * 000100000
 */
void breakpoint(int size,char *c,int* board,char *d)
{
  printf("%s\n", c);
  printf("%s%d:",d,size);
  for(int i=0;i<size;i++){
    if(board[i]==1){ printf("0"); }
    else if(board[i]==2){ printf("1"); }
    else if(board[i]==4){ printf("2"); }
    else if(board[i]==8){ printf("3"); }
    else if(board[i]==16){ printf("4"); }
    else if(board[i]==32){ printf("5"); }
    else if(board[i]==64){ printf("6"); }
    else if(board[i]==128){ printf("7"); }
    else if(board[i]==256){ printf("8"); }
    else if(board[i]==512){ printf("9"); }
    else if(board[i]==1024){ printf("10"); }
    else if(board[i]==-1){ printf("-"); }
  }
  printf("\n");
  //colの座標表示
  printf("   ");
  for (int j=0;j<size;j++){
    printf(" %2d",j);
  }
  printf("\n");
  printf(" =============");
  for (int j=0;j<size;j++){
    printf("==");
  }
  printf("\n");
  //row
  for (int j=0;j<size;j++){
    printf("%2d| ",j);
    if(board[j]==-1){ dec2bin(size,0); }
    else{ 
      dec2bin(size,board[j]); 
    }
    printf("\n");
  }
  printf("\n");
  /**
   *
   *
   */
  int moji;
	while ((moji = getchar()) != EOF){
		switch (moji){
		case '\n':
      printf("row:%d col:%d\n",board,board);
		  return;
		default:
			break;
		}
	}
}
/**
 *
 *
 */
void TimeFormat(clock_t utime,char* form)
{
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
/**
 * pressを枝刈りする
 * 角に置いて良いのは右上だけ
 * それ以外の角にクイーンを置いてもsymmetryOpsでスキップされる
 */
bool edakari_1(int size,int x,int y,int pa)
{
  if(pa==size-1||pa==0){
    if(x==0 && y==0){
    }else{
      return false; 
    }
  }
  return true;
}
/**
 * 角の場合は、次のpress_aの値は固定になる
 * 例えば （上）で1行目が右端だった場合、（右）の1行目は固定になる
 */
bool edakari_2(int size,int bpa,int pa)
{
  if(bpa==0){
    if(pa !=size-1){
      return false;
    }else{
      return true;
    }
  }
  return true;
}
/**
 * 角から2番目にクイーンを置いたら、次の2行目は0固定
 * 例:上1行目で左から2列目にクイーンを置いたら、左2行目は0固定
 */
bool edakari_3(int size,int bpa,int pb)
{
 if(bpa ==size-2){
  if(pb !=0){ return false; } 
 } 
 return true;
}
/**
 *上下左右２行２列にクイーンを配置する
 *
 */
bool board_placement(int size,int x,int y,struct Board* lb)
{
  long down,left,right;
  long TOPBIT=1<<(size-1);
  long SIDEMASK=(TOPBIT|1);
  long LASTMASK=SIDEMASK;
  for(int a=2;a<=lb->aBoard[0];a++){ LASTMASK|=LASTMASK>>1|LASTMASK<<1; }
  /**
   * １行目角にクイーンがある場合
   * 【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
   * ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
   */
  if(lb->aBoard[0]==0){
    if(y==1 && lb->aBoard[1]>x){
      return false;
    }  
  }else if(lb->aBoard[0] !=-1){
    /**
     * １行目角にクイーンがある場合以外
     * 【枝刈り】上部サイド枝刈り
     */
    if(x<lb->aBoard[0]){
      if((SIDEMASK&1<<y)==1){
        return false;
      }
    }
    /**
     * 【枝刈り】下部サイド枝刈り
     *
     */
    else if(x>size-1-lb->aBoard[0]){
      if((SIDEMASK&1<<y)==1){
        return false;
      }
    }
    /**
     * 【枝刈り】 最下段枝刈り
     *
     */
    if(x==size-1){
      if((LASTMASK&1<<y)!=0){
        return false;
      }
    }
  }
  /**
   * 同じ場所に置くかチェック
   *
   */
  if(lb->aBoard[x]==y){  
    return true;    //同じ場所に置くのはOK
  }
  //bv=1<<x;//xは行 yは列 p.N-1-x+yは右上から左下 x+yは左上から右下
  down=1<<y;
  lb->bBoard[x]=down;
  left=1<<(size-1-x+y);
  right=1<<(x+y);
  /**
   *left,down,rightの利き筋をチェックしてクイーンを置けるか判定する
   */
  if((lb->aBoard[x]!=-1)
   ||((lb->down&down)||(lb->left&left)||(lb->right&right))){
    return false;
  }     
  lb->aBoard[x]=y;
  lb->down|=down;
  lb->left|=left;
  lb->right|=right;
  return true;
}
/**
 *
 *
 */
int bit93_symmetryOps_n27(int size,struct Board* lb)
{
  int pressMinusTopSide;
  int press;
  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(lb->aBoard[0]==0){ return 2; }
  pressMinusTopSide=(size-2)*(size-1)-1-lb->topSide;
  press=(size-2)*(size-1)-1;
  /**
   *TODOここの枝刈り内容は不明
   */
  if(((lb->rightSide==pressMinusTopSide)&&(lb->leftSide<(press-lb->bottomSide)))
   ||((lb->bottomSide==pressMinusTopSide)&&(lb->leftSide>(press-lb->leftSide)))
   ||((lb->leftSide==pressMinusTopSide)&&(lb->bottomSide>(press-lb->rightSide)))){
    return 3;
  }
  /**
   *90度回転して同じかチェック
   */
  if(lb->rightSide==lb->topSide){
    if((lb->leftSide!=lb->topSide)||(lb->bottomSide!=lb->topSide)){ return 3; }
    return 0;
  }
  /**
   *180度回転して同じかチェック
   */
  if((lb->bottomSide==lb->topSide)&&(lb->leftSide>=lb->rightSide)){
    if(lb->leftSide>lb->rightSide){ return 3; }
    return 1;
  }
  /**
   *
   */
  return 2;   
}
/**
 *上から下に向かって1行ずつクイーンを置いていく
 *board_placementの処理で既にクイーンを置いている行はスキップする
 */
long bit93_countCompletions(int size,int row,int* aBoard,
    long left,long down,long right,int sym,int* bBoard)
{
  ecnt++;
  long bitmap,bit,cnt=0;
  while((aBoard[row]!=-1)&&(row<size)) { left<<=1; right>>=1; row++; }
  if(row==size){ return 1; }
  else{
    bitmap=~(left|down|right);   
    while(bitmap){
      bit=(-bitmap&bitmap);
      bitmap=(bitmap^bit);
      /**
       *sizeの数によってdownが数ビットスライドしているので
       *何列目にクイーンが置かれたのか知りたい場合はスライドした分を戻す
       */
      if(size==5){ bBoard[row]=(bit<<2); }
      else if(size==6){ bBoard[row]=(bit<<1); }
      else{ bBoard[row]=(bit>>(size-7)); }
      /**
       *
       *
       */
      breakpoint(size,"クイーンを配置",bBoard,"N");
      /**
       *
       *
       */
      cnt+=bit93_countCompletions(
          size,row+1,aBoard,
          (left|bit)<<1,down|bit,(right|bit)>>1,sym,bBoard);
    }
  }
  return cnt;
}
/**
 *
 *
 */
void symmetryOps(int size,struct Board* lBoard)
{
  /**
   *symmetryOpsで回転対称判定させ点数の低いものは枝刈りする
   *
   */
  lBoard->sym=bit93_symmetryOps_n27(size,lBoard);
  /**
   *symmetryOps突破したらcountCompletionsで上から下に各行1個ずつクイーンを置いていく
   *
   */
  if(lBoard->sym!=3){
    for(int j=0;j<=2;j++){ lBoard->COUNT[j]=0; }
    lBoard->row=2;
    lBoard->left=lBoard->left>>4;
    lBoard->down=((((lBoard->down>>2)|(~0<<(size-4)))+1)<<(size-5))-1;
    lBoard->right=(lBoard->right>>4)<<(size-5);
    lBoard->COUNT[lBoard->sym]+=bit93_countCompletions(
          size,lBoard->row,lBoard->aBoard,
          lBoard->left,lBoard->down,lBoard->right,lBoard->sym,lBoard->bBoard);
    /**
     *
     *
     */
    UNIQUE+=lBoard->COUNT[lBoard->sym];
    if(lBoard->sym==0){ TOTAL+=lBoard->COUNT[lBoard->sym]*2;}
    else if(lBoard->sym==1){ TOTAL+=lBoard->COUNT[lBoard->sym]*4; }
    else if(lBoard->sym==2){ TOTAL+=lBoard->COUNT[lBoard->sym]*8; }
  }
}
/**
 *2行だけ見た場合にクイーンを設置できる場所の組み合わせを作る
 *
 *
 */
void pres_idx(int size,int* pres_a,int* pres_b,int idx)
{
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      /**
       *
       *
       */
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){ continue; }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
}
/**
 *
 *
 *
 */
void bit93_NQueens(int size)
{
  /**
   *
   *
   */
  int pres_a[930],pres_b[930],idx=0;
  pres_idx(size,pres_a,pres_b,idx); 
  //
  //breakpoint(size,"クイーンを配置",bBoard,"N");
  
  /**
   * プログレス表示
   */
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);
  /**
   *上２行にクイーンを配置する
   *ミラーなので右側半分だけクイーンを設置する
   */
  for(int topSide=0;topSide<=(size/2)*(size-3);topSide++){
    /**
     *
     */
    printf("\r(%d/%d)",topSide,((size/2)*(size-3))); printf("\r"); fflush(stdout);
    /**
     *
     */
    int lsize=(size-2)*(size-1)-topSide;
    /**
     *
     */
    struct Board lBoard;
    lBoard.topSide=topSide;lBoard.down=lBoard.left=lBoard.right=0;
    for(int j=0;j<size;j++){ lBoard.aBoard[j]=-1;lBoard.bBoard[j]=-1; }
    if((!edakari_1(size,0,pres_a[topSide],pres_a[topSide]))
     ||(!board_placement(size,0,pres_a[topSide],&lBoard))
     ||(!board_placement(size,1,pres_b[topSide],&lBoard))){ 
      continue; 
    }
    /**
     *左側2行にクイーンを設置する
     *
     */
    struct Board leftSideB=lBoard;
    for(int leftSide=topSide;leftSide<lsize;leftSide++){
      lBoard=leftSideB;lBoard.leftSide=leftSide;
      if((!edakari_1(size,pres_a[leftSide],size-1,pres_a[leftSide]))
       ||(!edakari_3(size,pres_a[topSide],pres_b[leftSide]))
       ||(!board_placement(size,pres_a[leftSide],size-1,&lBoard))
       ||(!board_placement(size,pres_b[leftSide],size-2,&lBoard))){
        continue; 
      }
      /**
       *下側2行にクイーンを設置する
       *
       */
      struct Board bottomSideB=lBoard;
      for(int bottomSide=topSide;bottomSide<lsize;bottomSide++){
        lBoard=bottomSideB;lBoard.bottomSide=bottomSide;
        if((!edakari_1(size,size-1,size-1-pres_a[bottomSide],pres_a[bottomSide]))
         ||(!edakari_3(size,pres_a[leftSide],pres_b[bottomSide]))
         ||(!board_placement(size,size-1,size-1-pres_a[bottomSide],&lBoard))
         ||(!board_placement(size,size-2,size-1-pres_b[bottomSide],&lBoard))){
          continue; 
        }
        /**
         *右側2行にクイーンを設置する
         *
         */
        struct Board rightSideB=lBoard;
        for(int rightSide=topSide;rightSide<lsize;rightSide++){
          lBoard=rightSideB;
          lBoard.rightSide=rightSide;
          if((!edakari_1(size,size-1-pres_a[rightSide],0,pres_a[rightSide]))
           ||(!edakari_2(size,pres_a[topSide],pres_a[rightSide]))
           ||(!edakari_3(size,pres_a[topSide],pres_b[rightSide]))
           ||(!board_placement(size,size-1-pres_a[rightSide],0,&lBoard))
           ||(!board_placement(size,size-1-pres_b[rightSide],1,&lBoard))){ 
            continue; 
          }
          /**
           *上左下右2行2列にクイーンを置くことができたらsymmetryOpsで枝刈りする
           *symmetryOps抜けたらcountCompletionsで上から下に1行ずつクイーンを置いていく
           */
          symmetryOps(size,&lBoard);
        }
      } 
    }
  }
}
/**
 *
 *
 */
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,sgpu=false,q27=false;
  int argstart=1;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else if(argv[1][1]=='q'||argv[1][1]=='Q'){q27=true;}
    else{ cpur=true; }
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s]\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("  -q: q27 Version\n");
    printf("Default to 8 queen\n");
  }
  if(cpu){ printf("\n\n９−３．CPU 非再帰\n");
  }else if(cpur){ printf("\n\n９−３．CPUR 再帰\n");
  }else if(gpu){ printf( "\n\n９−３．GPU 非再帰\n");
  }else if(sgpu){ printf("\n\n９−３．SGPU 非再帰\n");
  }else if(q27){ printf( "\n\nQ２７．CPUR 再帰\n"); }
  if(cpu||cpur||q27){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;
    char t[20];
    int min=4;
    int targetN=19;
    for(int i=min;i<=targetN;i++){
      TOTAL=0; UNIQUE=0; ecnt=0; 
      st=clock();
      /**
       *
       */
      if(q27){ bit93_NQueens(i); }
      /**
       *
       */
      else{ bit93_NQueens(i); }
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s::%ld\n",i,TOTAL,UNIQUE,t,ecnt);
    }
  }
  return 0;
}
