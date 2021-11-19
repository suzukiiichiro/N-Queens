/**
 BITで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)


ブルートフォース


 コンパイルと実行
 $ gcc -O3 BIT05_N-Queen.cu && ./a.out -r 
                    -c:cpu 
                    -r cpu再帰 
                    -g GPU 
                    -s SGPU(サマーズ版と思われる)

    　 １．省略
    　 ２．：
    　 ３．：
    　 ４．：
  * 　 ５．ブルートフォース
    　 ６．right/leftの導入
    　 ７．対称解除法
    　 ８．動的分割統治法
    　 ９．クイーンの位置による分岐BOUND1
    １０．クイーンの位置による分岐BOUND1,2
    １１．枝刈り
    １２．最適化
    １３．並列処理

bash-5.1$ gcc -O3 BIT05_N-Queen.c && ./a.out -r
７．CPUR ブルートフォース

1: 上１列
00001
00000
00000
00000
00000

2: 上２列
00001
00001
00000
00000
00000
:
:
:
.
813799: 右１列
10001
10000
00000
00001
11001

813800: 右２列
10011
10000
00000
00001
11001

１、上下左右２列ずつにクイーンを配置する
上2行(0,0)(1,2) for(int w=0;w<=(size/2)*(size-3);w++){
左2列(3,6)(6,5) for(int n=w;n<lsize;n++){
下2行(6,5)(5,3) for(int e=w;e<lsize;e++){
右2列(0,0)(4,1) for(int s=w;s<lsize;s++){

(0,0)(6,5)は被っているので6個クイーンを置いた状態
0000001
0000100
0000000
1000000
0000010
0001000
0100000

２、回転チェックをして重複をスキップしたり、90度(*2),180度(*4),その他(*8)
に分類する

processに入る
３、クイーンを置いていく
B.bv 1111011
bvはどの行にクイーンが置かれているかを示している
0,1,3,4,5,6行目にクイーンが置かれていることがわかる
B.bh 1101111
bhはdown
0,1,2,3,5,6列目にクイーンが置かれていることがわかる
B.bu 1011111000
buはleft
B.bd 101100101001
bdはright

cnt[sym] += solve_nqueenr(B.bv >> 2,
  ¦ ¦ ((((B.bh>>2)|(~0<<(si-4)))+1)<<(si-5))-1,
  ¦ ¦ B.bu>>4,
  ¦ ¦ (B.bd>>4)<<(si-5));
すでに上２行はクイーンを置いているので、３行目からsolve_nqueenrをスタートする

B.bv >> 2 11110
bvは1行進むごとに右端を1ビットずつ削っていく
((((B.bh>>2)|(~0<<(si-4)))+1)<<(si-5))-1 down  1101111
B.bu>>4 left  右４ビット削除 101111
(B.bd>>4)<<(si-5)) 右４ビット削除後N-5個分左にシフト 1011001000

solve_nqueenr 1周目

if(bh+1==0){
全ての列にクイーンを置くことができると -1 1111111 となるので return 1して抜ける
bhは1101111なのでこのif文の中には入らない

while((bv&1)!=0) {
bv&1だと右端が1ということ
すでにクイーンが置かれていたらこの行ではクイーンを置かずleft,rightを1ビットシフトさせる。bvも右端を1ビット削る
bvは11110なので右端は0
３行目にはまだクイーンが置かれていないということになるのでこのif文の中には入らない

for(uint64 slots=~(bh|bu|bd);slots!=0;) {
slotsはクイーンの置ける場所。0だとどこにもクイーンが置けない
3行目はslots 0010000 なのでこのif文に入る
slot=slots&-slots;
slotは今回クイーンを置く場所
今回３行目は0010000 にクイーンを置く

0000001
0000100
0010000<---ここにクイーンを置く
1000000
0000010
0001000
0100000

solve_nqueenr 2周目
bv 1111
bh down 1111111
bu left 1111110
bd right 1101100

if(bh+1==0){
全ての列にクイーンを置くことができると -1 1111111 となるので return 1して抜ける
bhは1111111なのでこのif文に入って return 1して抜ける
**/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define THREAD_NUM 96
//#define THREAD_NUM		1
#define MAX 27
//
typedef unsigned long long uint64;
typedef struct
{
  uint64 bv;
  uint64 down;
  uint64 left;
  uint64 right;
  int x[MAX];
  int y[MAX];
} Board;
//
//関数宣言
void print(int size, char *c);
void dec2bin(int, int);
//
Board B;
unsigned int COUNT8 = 2;
unsigned int COUNT4 = 1;
unsigned int COUNT2 = 0;
long cnt[3];
long pre[3];
//変数宣言
long TOTAL = 0;   //GPU,CPUで使用
long UNIQUE = 0;  //GPU,CPUで使用
int DEBUG = true; //ボードレイアウト出力
int COUNT = 0;    //ボードレイアウト出力
//
void TimeFormat(clock_t utime, char *form)
{
  int dd, hh, mm;
  float ftime, ss;
  ftime = (float)utime / CLOCKS_PER_SEC;
  mm = (int)ftime / 60;
  ss = ftime - (int)(mm * 60);
  dd = mm / (24 * 60);
  mm = mm % (24 * 60);
  hh = mm / 60;
  mm = mm % 60;
  if (dd)
    sprintf(form, "%4d %02d:%02d:%05.2f", dd, hh, mm, ss);
  else if (hh)
    sprintf(form, "     %2d:%02d:%05.2f", hh, mm, ss);
  else if (mm)
    sprintf(form, "        %2d:%05.2f", mm, ss);
  else
    sprintf(form, "           %5.2f", ss);
}
//
long solve_nqueenr(uint64 bv, uint64 left, uint64 down, uint64 right)
{
  // Placement Complete?
  //printf("countCompletions_start\n");
  //printf("bv:%d\n",bv);
  //printf("bh:%d\n",bh);
  //printf("bu:%d\n",bu);
  //printf("bd:%d\n",bd);
  //bh=-1 1111111111 すべての列にクイーンを置けると-1になる
  if (down + 1 == 0)
  {
    //printf("return_bh+1==0:%d\n",bh);
    return 1;
  }
  // -> at least one more queen to place
  while ((bv & 1) != 0)
  { // Column is covered by pre-placement
    //bv 右端にクイーンがすでに置かれていたら。クイーンを置かずに１行下に移動する
    //bvを右端から１ビットずつ削っていく。ここではbvはすでにクイーンが置かれているかどうかだけで使う
    bv >>= 1;    //右に１ビットシフト
    left <<= 1;  //left 左に１ビットシフト
    right >>= 1; //right 右に１ビットシフト
    //printf("while:bv:%d\n",bv);
    //printf("while:bu:%d\n",bu);
    //printf("while:bd:%d\n",bd);
    //printf("while:bv&1:%d\n",bv&1);
  }
  //１行下に移動する
  bv >>= 1;
  //printf("onemore_bv:%d\n",bv);
  //printf("onemore_bh:%d\n",bh);
  //printf("onemore_bu:%d\n",bu);
  //printf("onemore_bd:%d\n",bd);
  //
  // Column needs to be placed
  long s = 0;
  uint64 bit;
  //bh:down bu:left bd:right
  //クイーンを置いていく
  //slotsはクイーンの置ける場所
  for (uint64 bitmap = ~(left | down | right); bitmap != 0; bitmap ^= bit)
  {
    //printf("colunm needs to be placed\n");
    //printf("slots:%d\n",slots);
    bit = bitmap & -bitmap;
    //printf("slot:%d\n",slot);
    //printf("bv:%d:bh|slot:%d:(bu|slot)<<1:%d:(bd|slot)>>1:%d\n",bv,bh|slot,(bu|slot)<<1,(bd|slot)>>1);
    s += solve_nqueenr(bv, (left | bit) << 1, down | bit, (right | bit) >> 1);
    //slots^=slot;
    //printf("slots:%d\n",slots);
  }
  //途中でクイーンを置くところがなくなるとここに来る
  //printf("return_cnt:%d\n",cnt);
  return s;
} // countCompletions()
//
void process(int si, Board B, int sym)
{
  //printf("process\n");
  pre[sym]++;
  //printf("N:%d\n",si);
  //BVは行 x
  //printf("getBV:%d\n",B.bv);
  //BHはdown y
  //printf("getBH:%d\n",B.bh);
  //BU left N-1-x+y 右上から左下
  //printf("getBU:%d\n",B.bu);
  //BD right x+y 左上から右下
  //printf("getBD:%d\n",B.bd);
  //printf("before_cnt_sym:%d\n",cnt[sym]);
  cnt[sym] += solve_nqueenr(B.bv >> 2,
                            B.left >> 4,
                            ((((B.down >> 2) | (~0 << (si - 4))) + 1) << (si - 5)) - 1,
                            (B.right >> 4) << (si - 5));

  //行 brd.getBV()>>2 右2ビット削除 すでに上２行はクイーンを置いているので進める BVは右端を１ビットずつ削っていく
  //列 down ((((brd.getBH()>>2)|(~0<<(N-4)))+1)<<(brd.N-5))-1 8だと左に1シフト 9:2 10:3
  //brd.getBU()>>4 left  右４ビット削除
  //(brd.getBD()>>4)<<(N-5)) right 右４ビット削除後N-5個分左にシフト
  //printf("cnt_sym:%d\n",cnt[sym]);
}
//
bool board_placement(int si, int x, int y)
{
  //同じ場所に置くかチェック
  //printf("i:%d:x:%d:y:%d\n",i,B.x[i],B.y[i]);
  if (B.x[x] == y)
  {
    //printf("Duplicate x:%d:y:%d\n",x,y);
    ////同じ場所に置くのはOK
    return true;
  }
  B.x[x] = y;
  //xは行 yは列 p.N-1-x+yは右上から左下 x+yは左上から右下
  uint64 bv = 1 << x;
  uint64 down = 1 << y;
  B.y[x] = B.y[x] + down;
  uint64 left = 1 << (si - 1 - x + y);
  uint64 right = 1 << (x + y);
  //printf("check valid x:%d:y:%d:p.N-1-x+y:%d;x+y:%d\n",x,y,si-1-x+y,x+y);
  //printf("check valid pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("bvcheck:%d:bhcheck:%d:bucheck:%d:bdcheck:%d\n",B.bv&bv,B.bh&bh,B.bu&bu,B.bd&bd);
  //if((B.bv&bv)||(B.down&down)||(B.left&left)||(B.right&right)){
  //printf("valid_false\n");
  //  return false;
  //}
  //printf("before pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  B.bv |= bv;
  B.down |= down;
  B.left |= left;
  B.right |= right;
  //printf("after pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("valid_true\n");
  return true;
}
//CPUR 再帰版 ロジックメソッド
void _NQueenR(int size)
{
  int pres_a[930];
  int pres_b[930];
  int idx = 0;
  for (int a = 0; a < size; a++)
  {
    for (int b = 0; b < size; b++)
    {
      if ((a >= b && (a - b) <= 1) || (b > a && (b - a) <= 1))
      {
        continue;
      }
      pres_a[idx] = a;
      pres_b[idx] = b;
      idx++;
    }
  }
  //プログレス
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)", (unsigned)pres_a[(size / 2) * (size - 3)], (unsigned)pres_b[(size / 2) * (size - 3)], (unsigned)pres_a[(size / 2) * (size - 3) + 1], (unsigned)pres_b[(size / 2) * (size - 3) + 1]);

  Board wB = B;
  for (int w = 0; w <= (size / 2) * (size - 3); w++)
  {
    //上２行にクイーンを置く
    //上１行は２分の１だけ実行
    //q=7なら (7/2)*(7-4)=12
    //1行目は0,1,2で,2行目0,1,2,3,4,5,6 で利き筋を置かないと13パターンになる

    B = wB;
    //
    // B.bv=0;
    // B.bh=0;
    // B.bu=0;
    // B.bd=0;
    B.bv = B.down = B.left = B.right = 0;
    //
    for (int i = 0; i < size; i++)
    {
      B.x[i] = -1;
    }
    // 不要
    // int wa=pres_a[w];
    // int wb=pres_b[w];
    //
    //プログレス
    printf("\r(%d/%d)", w, ((size / 2) * (size - 3))); // << std::flush;
    printf("\r");
    fflush(stdout);

    //上２行　0行目,1行目にクイーンを置く
    //
    // 謎１
    //
    // 置き換え
    // board_placement(size,0,wa);
    //0行目にクイーンを置く
    // board_placement(size,x,y)
    board_placement(size, 0, pres_a[w]);
    //printf("placement_pwa:x:0:y:%d\n",pres_a[w]);
    //
    //
    // 謎２
    //
    //置き換え
    //board_placement(size,1,wb);
    //1行目にクイーンを置く
    // board_placement(size,x,y)
    board_placement(size, 1, pres_b[w]);
    //printf("placement_pwb:x:1:y:%d\n",pres_b[w]);

    Board nB = B;
    //追加
    int lsize = (size - 2) * (size - 1) - w;
    //for(int n=w;n<(size-2)*(size-1)-w;n++){
    for (int n = w; n < lsize; n++)
    {
      //左２列にクイーンを置く
      B = nB;
      //printf("nloop:n:%d\n",n);
      //
      // 不要
      // int na=pres_a[n];
      // int nb=pres_b[n];
      //
      //置き換え
      //bool pna=board_placement(size,na,size-1);
      //bool pna=board_placement(size,pres_a[n],size-1);
      //インライン
      //if(pna==false){
      if (board_placement(size, pres_a[n], size - 1) == false)
      {
        //printf("pnaskip:na:%d:N-1:%d\n",na,size-1);
        continue;
      }
      //printf("placement_pna:x:%d:yk(N-1):%d\n",pres_a[n],size-1);
      //置き換え
      //bool pnb=board_placement(size,nb,size-2);
      //bool pnb=board_placement(size,pres_b[n],size-2);
      //インライン
      //if(pnb==false){
      if (board_placement(size, pres_b[n], size - 2) == false)
      {
        //printf("pnbskip:nb:%d:N-2:%d\n",nb,size-2);
        continue;
      }
      //printf("placement_pnb:x:%d:yk(N-2):%d\n",pres_b[n],size-2);
      Board eB = B;
      //for(int e=w;e<(size-2)*(size-1)-w;e++){
      for (int e = w; e < lsize; e++)
      {
        //下２行に置く
        B = eB;
        //printf("eloop:e:%d\n",e);
        //不要
        //int ea=pres_a[e];
        //int eb=pres_b[e];
        //置き換え
        //bool pea=board_placement(size,size-1,size-1-ea);
        //インライン
        //if(pea==false){
        if (board_placement(size, size - 1, size - 1 - pres_a[e]) == false)
        {
          //printf("peaskip:N-1:%d:N-1-ea:%d\n",size-1,size-1-ea);
          continue;
        }
        //printf("placement_pea:xk(N-1):%d:y:%d\n",size-1,size-1-pres_a[e]);
        //置き換え
        //bool peb=board_placement(size,size-2,size-1-eb);
        //インライン
        //if(peb==false){
        if (board_placement(size, size - 2, size - 1 - pres_b[e]) == false)
        {
          //printf("pebskip:N-2:%d:N-1-eb:%d\n",size-2,size-1-eb);
          continue;
        }
        //printf("placement_peb:xk(N-2):%d:y:%d\n",size-2,size-1-pres_b[e]);
        Board sB = B;
        //for(int s=w;s<(size-2)*(size-1)-w;s++){
        for (int s = w; s < lsize; s++)
        {
          ////右２列に置く
          B = sB;
          //printf("sloop:s:%d\n",s);
          //
          //不要
          //int sa =pres_a[s];
          //int sb =pres_b[s];
          //
          //置き換え
          //bool psa=board_placement(size,size-1-sa,0);
          //インライン
          //if(psa==false){
          if (board_placement(size, size - 1 - pres_a[s], 0) == false)
          {
            //printf("psaskip:N-1-sa:%d:0\n",size-1-sa);
            continue;
          }
          //printf("psa:x:%d:yk(0):0\n",size-1-pres_a[s]);
          //bool psb=board_placement(size,size-1-sb,1);
          //if(psb==false){
          if (board_placement(size, size - 1 - pres_b[s], 1) == false)
          {
            //printf("psbskip:N-1-sb:%d:1\n",size-1-sb);
            continue;
          }
          //printf("psb:x:%d:yk(1):1\n",size-1-pres_b[s]);
          //printf("noskip\n");
          //printf("pwa:xk(0):0:y:%d\n",pres_a[w]);
          //printf("pwb:xk(1):1:y:%d\n",pres_b[w]);
          //printf("pna:x:%d:yk(N-1):%d\n",pres_a[n],size-1);
          //printf("pnb:x:%d:yk(N-2):%d\n",pres_b[n],size-2);
          //printf("pea:xk(N-1):%d:y:%d\n",size-1,size-1-pres_a[e]);
          //printf("peb:xk(N-2):%d:y:%d\n",size-2,size-1-pres_b[e]);
          //printf("psa:x:%d:yk(0):0\n",size-1-pres_a[s]);
          //printf("psb:x:%d:yk(1):1\n",size-1-pres_b[s]);
          //
          //// Check for minimum if n, e, s = (N-2)*(N-1)-1-w
          int ww = (size - 2) * (size - 1) - 1 - w;
          //新設
          int w2 = (size - 2) * (size - 1) - 1;
          //if(s==ww){
          if ((s == ww) && (n < (w2 - e)))
          {
            //check if flip about the up diagonal is smaller
            //if(n<(size-2)*(size-1)-1-e){
            //if(n<(w2-e)){
            continue;
            //}
          }
          //if(e==ww){
          if ((e == ww) && (n > (w2 - n)))
          {
            //check if flip about the vertical center is smaller
            //if(n>(size-2)*(size-1)-1-n){
            //if(n>(w2-n)){
            continue;
            //}
          }
          //if(n==ww){
          if ((n == ww) && (e > (w2 - s)))
          {
            //// check if flip about the down diagonal is smaller
            //if(e>(size-2)*(size-1)-1-s){
            //if(e>(w2-s)){
            continue;
            //}
          }
          //// Check for minimum if n, e, s = w
          if (s == w)
          {
            if ((n != w) || (e != w))
            {
              // right rotation is smaller unless  w = n = e = s
              //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
              continue;
            }
            //w=n=e=sであれば90度回転で同じ可能性
            //この場合はミラーの2
            process(size, B, COUNT2);
            //(*act)(board, Symmetry::ROTATE);
            continue;
          }
          if ((e == w) && (n >= s))
          {
            //if(n>=s){
            //e==wは180度回転して同じ
            if (n > s)
            {
              //180度回転して同じ時n>=sの時はsmaller?
              continue;
            }
            //この場合は4
            process(size, B, COUNT4);
            //(*act)(board, Symmetry::POINT);
            continue;
            //}
          }
          process(size, B, COUNT8);
          //(*act)(board, Symmetry::NONE);
          //この場合は8
          continue;
        }
      }
    }
  }
  //printf("ROTATE_0:%d\n",cnt[ROTATE]);
  //printf("POINT_1:%d\n",cnt[POINT]);
  //printf("NONE_2:%d\n",cnt[NONE]);
  UNIQUE = cnt[COUNT2] + cnt[COUNT4] + cnt[COUNT8];
  TOTAL = cnt[COUNT2] * 2 + cnt[COUNT4] * 4 + cnt[COUNT8] * 8;
}
//出力
void dec2bin(int size, int dec)
{
  int i, b[32];
  for (i = 0; i < size; i++)
  {
    b[i] = dec % 2;
    dec = dec / 2;
  }
  while (i > 0)
    printf("%1d", b[--i]);
}
void print(int size, char *c)
{
  printf("%d: %s\n", ++COUNT, c);
  for (int j = 0; j < size; j++)
  {
    dec2bin(size, B.y[j]);
    printf("\n");
  }
  printf("\n");
}
//
void NQueenR(int size)
{
  int pres_a[930];
  int pres_b[930];
  int idx = 0;
  for (int a = 0; a < size; a++)
  {
    for (int b = 0; b < size; b++)
    {
      //if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){
      //  continue;
      //}
      pres_a[idx] = a;
      pres_b[idx] = b;
      idx++;
    }
  }
  Board wB = B;
  for (int w = 0; w < idx; w++)
  { //5で修正
    //for(int w=0;w<size*size;w++){
    //
    //N=5 の場合
    //for(int w=0;w<=(size/2)*(size-3);w++){
    //
    //上２行にクイーンを配置できるパターン数
    //
    // 4 3 2 1 0
    // ----------
    // x x x 0 0 |0
    // 0 0 0 0 0 |1
    //
    //xxx00
    //00000
    //
    // N=5 の場合
    //(5/2)*(5-3)=4
    //1行目は0,1で,2行目0,1,2,3,4で利き筋を
    //考慮すると0から4までなので５パターン
    //
    //ミラーにより、後で２倍する関係で、
    // １行目は半分しかクイーンを置かない
    //N=5 の場合は、１行目は右から１番目、２番目に
    //だけクイーンを置く
    //
    //クイーンを１行目右端に置く場合
    //２行目には右から３、４、５番目に置ける
    //(0,0:1,2)(0,0:1,3)(0,0:1,4)
    //
    // 1パターン
    // 4 3 2 1 0
    // ----------
    // x x x x 0 |0
    // x x 0 x x |1
    //
    // 2パターン
    // 4 3 2 1 0
    // ----------
    // x x x x 0 |0
    // x 0 x x x |1
    //
    // 3パターン
    // 4 3 2 1 0
    // ----------
    // x x x x 0 |0
    // 0 x x x x |1
    //
    //クイーンを１行目右から２番目に置く場合
    //２行目には右から４、５番目に置ける
    //(0,1:1,3)(0,1:1,4)
    //
    // 5パターン
    // 4 3 2 1 0
    // ----------
    // x x x 0 x |0
    // x 0 x x x |1
    //
    //
    // 5パターン
    // 4 3 2 1 0
    // ----------
    // x x x 0 x |0
    // 0 x x x x |1
    //
    //
    //
    B = wB;
    B.bv = B.down = B.left = B.right = 0;
    for (int i = 0; i < size; i++)
    {
      B.x[i] = -1;
    }
    //上２列に置く
    board_placement(size, 0, pres_a[w]);
    if (DEBUG)
    {
      print(size, "上１列");
    }
    board_placement(size, 1, pres_b[w]);
    if (DEBUG)
    {
      print(size, "上２列");
    }
    Board nB = B;
    int lsize = (size - 2) * (size - 1) - w;
    for (int n = 0; n < idx; n++)
    { //6で修正
      //左２列に置く
      B = nB;
      if (board_placement(size, pres_a[n], size - 1) == false)
      {
        continue;
      }
      if (DEBUG)
      {
        print(size, "左１列");
      }
      if (board_placement(size, pres_b[n], size - 2) == false)
      {
        continue;
      }
      if (DEBUG)
      {
        print(size, "左２列");
      }
      Board eB = B;
      for (int e = 0; e < idx; e++)
      { //6で修正
        //下２行に置く
        B = eB;
        if (board_placement(size, size - 1, size - 1 - pres_a[e]) == false)
        {
          continue;
        }
        if (DEBUG)
        {
          print(size, "下１列");
        }
        if (board_placement(size, size - 2, size - 1 - pres_b[e]) == false)
        {
          continue;
        }
        if (DEBUG)
        {
          print(size, "下２列");
        }
        //右２列に置く
        Board sB = B;
        for (int s = 0; s < idx; s++)
        { //6で修正
          B = sB;
          if (board_placement(size, size - 1 - pres_a[s], 0) == false)
          {
            continue;
          }
          if (DEBUG)
          {
            print(size, "右１列");
          }
          if (board_placement(size, size - 1 - pres_b[s], 1) == false)
          {
            continue;
          }
          if (DEBUG)
          {
            print(size, "右２列");
          }
          //対称解除法
          //int ww=(size-2)*(size-1)-1-w;//5で修正
          //int w2=(size-2)*(size-1)-1;//5で修正
          //if((s==ww)&&(n<(w2-e))){ continue; }//6で修正
          //if((e==ww)&&(n>(w2-n))){ continue; }//6で修正
          //if((n==ww)&&(e>(w2-s))){ continue; }//6で修正
          //if(s==w){ if((n!=w)||(e!=w)){ continue; }//6で修正
          process(size, B, COUNT2);
          continue;
          //}//5で修正
          //if((e==w)&&(n>=s)){//6で修正
          //  if(n>s){ continue; } //6で修正
          //  process(size,B,COUNT4); continue; //6で修正
          //}//5で修正
          //process(size,B,COUNT8); continue;//6で修正
        }
      }
    }
  }
  //TOTAL=UNIQUE=cnt[COUNT2]+cnt[COUNT4]+cnt[COUNT8];//6で修正
  //TOTAL=cnt[COUNT2]*2+cnt[COUNT4]*4+cnt[COUNT8]*8;//6で修正
}
//メインメソッド
int main(int argc, char **argv)
{
  bool cpu = false, cpur = false, gpu = false, sgpu = false;
  int argstart = 1;
  //int steps=24576;

  /** パラメータの処理 */
  if (argc >= 2 && argv[1][0] == '-')
  {
    if (argv[1][1] == 'c' || argv[1][1] == 'C')
    {
      cpu = true;
    }
    else if (argv[1][1] == 'r' || argv[1][1] == 'R')
    {
      cpur = true;
    }
    else if (argv[1][1] == 'g' || argv[1][1] == 'G')
    {
      gpu = true;
    }
    else if (argv[1][1] == 's' || argv[1][1] == 'S')
    {
      sgpu = true;
    }
    else
      cpur = true;
    argstart = 2;
  }
  if (argc < argstart)
  {
    printf("Usage: %s [-c|-g|-r|-s]\n", argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if (cpu)
  {
    printf("\n\n７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }
  else if (cpur)
  {
    printf("\n\n７．CPUR ブルートフォース\n");
  }
  else if (gpu)
  {
    printf("\n\n７．GPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
  }
  else if (sgpu)
  {
    printf("\n\n７．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if (cpu || cpur)
  {
    printf("%s\n", " N:        Total       Unique        hh:mm:ss.ms");
    clock_t st; //速度計測用
    char t[20]; //hh:mm:ss.msを格納
    int min=5; int targetN=17;
    //int mask;
    for (int i = min; i <= targetN; i++)
    {
      /***07 symmetryOps CPU,GPU同一化*********************/
      TOTAL = 0;
      UNIQUE = 0;
      //COUNT2=COUNT4=COUNT8=0;
      for (int j = 0; j <= 2; j++)
      {
        pre[j] = 0;
        cnt[j] = 0;
      }
      /************************/
      //mask=(1<<i)-1;
      st = clock();
      //
      //【通常版】
      //if(cpur){ _NQueenR(i,mask,0,0,0,0); }
      //CPUR
      if (cpur)
      {
        NQueenR(i);
        //printf("通常版\n");
      }
      //CPU
      if (cpu)
      {
        //NQueen(i,mask);
        printf("準備中\n");
      }
      //
      TimeFormat(clock() - st, t);
      /***07 symmetryOps CPU,GPU同一化*********************/
      printf("%2d:%13ld%16ld%s\n", i, TOTAL, UNIQUE, t);
      /************************/
    }
  }
  if (gpu || sgpu)
  {
    int min = 5;
    int targetN = 5;
    //int min=8;int targetN=8;

    struct timeval t0;
    struct timeval t1;
    int ss;
    int ms;
    int dd;
    printf("%s\n", " N:        Total      Unique      dd:hh:mm:ss.ms");
    for (int i = min; i <= targetN; i++)
    {
      gettimeofday(&t0, NULL); // 計測開始
      if (gpu)
      {
        TOTAL = 0;
        UNIQUE = 0;
        //NQueenG(i,steps);
      }
      else if (sgpu)
      {
        printf("準備中");
        //TOTAL=sgpu_solve_nqueen_cuda(i,steps);
      }
      gettimeofday(&t1, NULL); // 計測終了
      if (t1.tv_usec < t0.tv_usec)
      {
        dd = (int)(t1.tv_sec - t0.tv_sec - 1) / 86400;
        ss = (t1.tv_sec - t0.tv_sec - 1) % 86400;
        ms = (1000000 + t1.tv_usec - t0.tv_usec + 500) / 10000;
      }
      else
      {
        dd = (int)(t1.tv_sec - t0.tv_sec) / 86400;
        ss = (t1.tv_sec - t0.tv_sec) % 86400;
        ms = (t1.tv_usec - t0.tv_usec + 500) / 10000;
      }
      int hh = ss / 3600;
      int mm = (ss - hh * 3600) / 60;
      ss %= 60;
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n", i, TOTAL, UNIQUE, dd, hh, mm, ss, ms);
    }
  }
  return 0;
}