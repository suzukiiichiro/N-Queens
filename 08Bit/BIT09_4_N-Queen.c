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
    ->90度回転、180度回転させて同じになることはないので正解がある場合は必ず88
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
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
long TOTAL=0;
long UNIQUE=0;
typedef struct{
  int row;
  int sym;
  int bv;
  long down;
  long left;
  long right;
  int aBoard[MAX];
  int topSide;
  int leftSide;
  int bottomSide;
  int rightSide;
  int BOUND2;
  int TOPBIT;
  int SIDEMASK;
  int LASTMASK;
  long COUNT[3];
}Board ;

void TimeFormat(clock_t utime,char *form)
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
bool board_placement(int size,int x,int y,Board *lb)
{
  int bv;
  long down;
  long left;
  long right;

  //１行目角にクイーンがある場合
  //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
  // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
  if(lb->aBoard[0]==0){
    if(y==1 && lb->aBoard[1]>x){
      return false;
    }  
  }else if(lb->aBoard[0] !=-1){
  //１行目角にクイーンがある場合以外
	//【枝刈り】上部サイド枝刈り
    if(x<lb->aBoard[0]){
      if((lb->SIDEMASK&1<<y)==1){
        return false;
      }
    }else if(x>lb->BOUND2){
  //【枝刈り】下部サイド枝刈り
      if((lb->SIDEMASK&1<<y)==1){
        return false;
      }
    }
  //【枝刈り】 最下段枝刈り
    if(x==size-1){
      if((lb->LASTMASK&1<<y)!=0){
        return false;
      }
    }
  }
  if(lb->aBoard[x]==y){  //同じ場所に置くかチェック
    return true;    //同じ場所に置くのはOK
  }
  lb->aBoard[x]=y;
  bv=1<<x;//xは行 yは列 p.N-1-x+yは右上から左下 x+yは左上から右下
  down=1<<y;
  left=1<<(size-1-x+y);
  right=1<<(x+y);
  if((lb->bv&bv)||(lb->down&down)||(lb->left&left)||(lb->right&right)){
    return false;
  }     
  lb->bv|=bv;
  lb->down|=down;
  lb->left|=left;
  lb->right|=right;
  return true;
}
int symmetryOps_n27(int w,int e,int n,int s,int size)
{
  int ww;
  int w2;
  //
  //// Check for minimum if n, e, s = (N-2)*(N-1)-1-w
  ww=(size-2)*(size-1)-1-w;
  //新設
  w2=(size-2)*(size-1)-1;
  //if(s==ww){
  if((s==ww)&&(n<(w2-e))){
    //check if flip about the up diagonal is smaller
    //if(n<(size-2)*(size-1)-1-e){
    //if(n<(w2-e)){
    return 3;
    //}
  }
  //if(e==ww){
  if((e==ww)&&(n>(w2-n))){
    //check if flip about the vertical center is smaller
    //if(n>(size-2)*(size-1)-1-n){
    //if(n>(w2-n)){
    return 3;       
    //}
  }
  //if(n==ww){
  if((n==ww)&&(e>(w2-s))){
    //// check if flip about the down diagonal is smaller
    //if(e>(size-2)*(size-1)-1-s){
    //if(e>(w2-s)){
    return 3;
    //}
  }
  if(s==w){
    if((n!=w)||(e!=w)){
      // right rotation is smaller unless  w = n = e = s
      //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
      return 3;
    }
    //w=n=e=sであれば90度回転で同じ可能性
    //この場合はミラーの2
    return 0;
  }
  if((e==w)&&(n>=s)){
    //e==wは180度回転して同じ
    if(n>s){
      //180度回転して同じ時n>=sの時はsmaller?
      return 3;
    }
    //この場合は4
    return 1;
  }
  return 2;   
}
int bit93_symmetryOps_n27(int size,Board* lb)
{
  int pressMinusTopSide;
  int press;

  //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
  if(lb->aBoard[0]==0){
    return 2; 
  }
  //// Check for minimum if n, e, s = (N-2)*(N-1)-1-w
  pressMinusTopSide=(size-2)*(size-1)-1-lb->topSide;
  //新設
  press=(size-2)*(size-1)-1;
  if((lb->rightSide==pressMinusTopSide)&&(lb->leftSide<(press-lb->bottomSide))){
    //check if flip about the up diagonal is smaller
    return 3;
  }
  if((lb->bottomSide==pressMinusTopSide)&&(lb->leftSide>(press-lb->leftSide))){
    return 3;       
  }
  if((lb->leftSide==pressMinusTopSide)&&(lb->bottomSide>(press-lb->rightSide))){
    return 3;
  }
  if(lb->rightSide==lb->topSide){
    if((lb->leftSide!=lb->topSide)||(lb->bottomSide!=lb->topSide)){
      // right rotation is smaller unless  w = n = e = s
      //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
      return 3;
    }
    //w=n=e=sであれば90度回転で同じ可能性
    //この場合はミラーの2
    return 0;
  }
  if((lb->bottomSide==lb->topSide)&&(lb->leftSide>=lb->rightSide)){
    //e==wは180度回転して同じ
    //if(leftSide>rightSide){
    if(lb->leftSide>lb->rightSide){
      //180度回転して同じ時n>=sの時はsmaller?
      return 3;
    }
    //この場合は4
    return 1;
  }
  return 2;   
}
void q27_countCompletions(int bv,long down,long left,long right,Board* lb,int sym)
{
  if(down+1 == 0){
    lb->COUNT[sym]++;
  }
  while((bv&1) != 0) { // Column is covered by pre-placement
    bv >>= 1;//右に１ビットシフト
    left <<= 1;//left 左に１ビットシフト
    right >>= 1;//right 右に１ビットシフト
  }
  bv >>= 1;//１行下に移動する
  //bh:down bu:left bd:right
  //クイーンを置いていく
  //slotsはクイーンの置ける場所
  for(long bitmap = ~(down|left|right); bitmap != 0;) {
    long const  bit = bitmap & -bitmap;
    q27_countCompletions(bv, down|bit, (left|bit) << 1, (right|bit) >> 1,lb,sym);
    bitmap ^= bit;
  }
  //途中でクイーンを置くところがなくなるとここに来る
  //printf("return_cnt:%d\n",cnt);
}
long bit93_countCompletions1(int size,int row,int bv,long left,long down,long right,int sym)
{
  long bitmap=0;
  long bit=0;
  //
  //既にクイーンを置いている行はスキップする
  while((bv&1)!=0) {
    bv>>=1;//右に１ビットシフト
    left<<=1;//left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト  
    row++; 
  }
  bv>>=1;
  long  cnt = 0;
  if(row==size){
    //TOTAL++;
      //UNIQUE++;       //ユニーク解を加算
      //TOTAL+=cnt;       //対称解除で得られた解数を加算
      return 1;
  }else{
      //bitmap=mask&~(left|down|right);//maskつけると10桁目以降数が出なくなるので外した
      bitmap=~(left|down|right);   
      while(bitmap>0){
          bit=(-bitmap&bitmap);
          bitmap=(bitmap^bit);
          cnt+=bit93_countCompletions1(size,row+1,bv,(left|bit)<<1,down|bit,(right|bit)>>1,sym);
      }
  }
  return cnt;
}
long bit93_countCompletions2(int size,int row,int bv,long left,long down,long right,int sym)
{
  long bitmap=0;
  long bit=0;
  //
  //既にクイーンを置いている行はスキップする
  while((bv&1)!=0) {
    bv>>=1;//右に１ビットシフト
    left<<=1;//left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト  
    row++; 
  }
  bv>>=1;
  long  cnt = 0;
  if(row==size){
    //TOTAL++;
      //UNIQUE++;       //ユニーク解を加算
      //TOTAL+=cnt;       //対称解除で得られた解数を加算
      return 1;
  }else{
      //bitmap=mask&~(left|down|right);//maskつけると10桁目以降数が出なくなるので外した
      bitmap=~(left|down|right);   
      while(bitmap>0){
          bit=(-bitmap&bitmap);
          bitmap=(bitmap^bit);
          cnt+=bit93_countCompletions2(size,row+1,bv,(left|bit)<<1,down|bit,(right|bit)>>1,sym);
      }
  }
  return cnt;
}
void q27_process(int size,Board* lb,int sym)
{
  q27_countCompletions(lb->bv >> 2,
    ((((lb->down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
    lb->left>>4,(lb->right>>4)<<(size-5),lb,sym);
}
void bit93_NQueens(int size)
{
  Board lBoard;
  Board leftSideB,rightSideB,bottomSideB,topSideB;

  int pres_a[930];
  int pres_b[930];
  int idx=0;
  int lsize;
  int sym;
  

  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){ continue; }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //プログレス
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);
  //プログレス
  int TOPBIT=1<<(size-1);
  int SIDEMASK=(TOPBIT|1);
  int LASTMASK=SIDEMASK;
  int BOUND2=size-2;
  int beforepres=0;
  topSideB=lBoard;
  for(int topSide=0;topSide<=(size/2)*(size-3);topSide++){
    //プログレス
    printf("\r(%d/%d)",topSide,((size/2)*(size-3)));// << std::flush;
    printf("\r");
    fflush(stdout);
    //プログレス
    if(pres_a[topSide]>1 && pres_a[topSide]>beforepres){
      
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      BOUND2=size-1-pres_a[topSide];
      beforepres=pres_a[topSide];
    }
    lBoard=topSideB;
    lBoard.bv=lBoard.down=lBoard.left=lBoard.right=0;
    lBoard.SIDEMASK=SIDEMASK;
    lBoard.LASTMASK=LASTMASK;
    lBoard.BOUND2=BOUND2;
    for(int j=0;j<size;j++){ lBoard.aBoard[j]=-1; }
    board_placement(size,0,pres_a[topSide],&lBoard);
    if(board_placement(size,1,pres_b[topSide],&lBoard)==false){
      continue;
    }
    leftSideB=lBoard;
    lsize=(size-2)*(size-1)-topSide;
    for(int leftSide=topSide;leftSide<lsize;leftSide++){
      lBoard=leftSideB;
      if(board_placement(size,pres_a[leftSide],size-1,&lBoard)==false){ 
        continue; 
      }
      if(board_placement(size,pres_b[leftSide],size-2,&lBoard)==false){ 
        continue; 
      }
      bottomSideB=lBoard;
      for(int bottomSide=topSide;bottomSide<lsize;bottomSide++){
        lBoard=bottomSideB;  
        if(board_placement(size,size-1,size-1-pres_a[bottomSide],&lBoard)==false){
          continue; 
        }
        if(board_placement(size,size-2,size-1-pres_b[bottomSide],&lBoard)==false){
          continue;
        }
        rightSideB=lBoard;
        for(int rightSide=topSide;rightSide<lsize;rightSide++){
          lBoard=rightSideB;
          if(board_placement(size,size-1-pres_a[rightSide],0,&lBoard)==false){
            continue; 
          }
          if(board_placement(size,size-1-pres_b[rightSide],1,&lBoard)==false){
            continue; 
          }
          lBoard.topSide=topSide;
          lBoard.bottomSide=bottomSide;
          lBoard.leftSide=leftSide;
          lBoard.rightSide=rightSide;
          sym=bit93_symmetryOps_n27(size,&lBoard);
          if(sym !=3){
            for(int j=0;j<=2;j++){ lBoard.COUNT[j]=0; }
            //NQueenの処理
            lBoard.row=2;
            lBoard.sym=sym;
            lBoard.bv=lBoard.bv>>2;
            lBoard.left=lBoard.left>>4;
            lBoard.down=((((lBoard.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1;
            lBoard.right=(lBoard.right>>4)<<(size-5);
            //１行目のQが角か角以外かで分岐する
            if(lBoard.aBoard[0]==0){
              lBoard.COUNT[sym]+=bit93_countCompletions1(size,lBoard.row,lBoard.bv,lBoard.left,lBoard.down,lBoard.right,lBoard.sym);
            }else{
              lBoard.COUNT[sym]+=bit93_countCompletions2(size,lBoard.row,lBoard.bv,lBoard.left,lBoard.down,lBoard.right,lBoard.sym);
            }
            UNIQUE+=lBoard.COUNT[sym];
            if(sym==0){ TOTAL+=lBoard.COUNT[sym]*2; }
            else if(sym==1){ TOTAL+=lBoard.COUNT[sym]*4; }
            else if(sym==2){ TOTAL+=lBoard.COUNT[sym]*8; }
          }
        }
      } 
    }
  }
}
void q27_NQueens(int size)
{
  Board lBoard;
  Board nB,sB,eB,wB;

  int pres_a[930];
  int pres_b[930];
  int idx=0;
  int lsize;
  int sym;

  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){ continue; }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  //プログレス
  printf("\t\t  First side bound: (%d,%d)/(%d,%d)",(unsigned)pres_a[(size/2)*(size-3)  ],(unsigned)pres_b[(size/2)*(size-3)  ],(unsigned)pres_a[(size/2)*(size-3)+1],(unsigned)pres_b[(size/2)*(size-3)+1]);
  //プログレス
  wB=lBoard;
  for (int w = 0; w <= (size / 2) * (size - 3); w++){
    //プログレス
    printf("\r(%d/%d)",w,((size/2)*(size-3)));// << std::flush;
    printf("\r");
    fflush(stdout);
    //プログレス
    lBoard=wB;
    lBoard.bv=lBoard.down=lBoard.left=lBoard.right=0;
    for(int j=0;j<size;j++){ lBoard.aBoard[j]=-1; }
    board_placement(size,0,pres_a[w],&lBoard);
    board_placement(size,1,pres_b[w],&lBoard);
    nB=lBoard;
    lsize=(size-2)*(size-1)-w;
    for(int n=w;n<lsize;n++){
      lBoard=nB;
      if(board_placement(size,pres_a[n],size-1,&lBoard)==false){ 
        continue; 
      }
      if(board_placement(size,pres_b[n],size-2,&lBoard)==false){ 
        continue; 
      }
      eB=lBoard;
      for(int e=w;e<lsize;e++){
        lBoard=eB;  
        if(board_placement(size,size-1,size-1-pres_a[e],&lBoard)==false){ 
          continue; 
        }
        if(board_placement(size,size-2,size-1-pres_b[e],&lBoard)==false){ 
          continue; 
        }
        sB=lBoard;
        for(int s=w;s<lsize;s++){
          lBoard=sB;
          if(board_placement(size,size-1-pres_a[s],0,&lBoard)==false){ 
            continue; 
          }
          if(board_placement(size,size-1-pres_b[s],1,&lBoard)==false){ 
            continue; 
          }
          sym=symmetryOps_n27(w,e,n,s,size);
          if(sym !=3){
            for(int j=0;j<=2;j++){ lBoard.COUNT[j]=0; }
            q27_process(size,&lBoard,sym);
            UNIQUE+=lBoard.COUNT[sym];
            if(sym==0){ TOTAL+=lBoard.COUNT[sym]*2; }
            else if(sym==1){ TOTAL+=lBoard.COUNT[sym]*4; }
            else if(sym==2){ TOTAL+=lBoard.COUNT[sym]*8; }
          }
        }
      } 
    }
  }
}
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,sgpu=false,q27=false;
  int argstart=1;
  //,steps=24576;
  /** パラメータの処理 */
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
    else if(argv[1][1]=='q'||argv[1][1]=='Q'){q27=true;}
    else{ cpur=true;
    }
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
  if(cpu){
    printf("\n\n９−３．CPU 非再帰 ビット + 対象解除 + BackTrack1 ＋ BackTrack2 + 枝刈り\n");
  }else if(cpur){
    printf("\n\n９−３．CPUR 再帰 ビット + 対象解除 + BackTrack1 ＋ BackTrack2 + 枝刈り\n");
  }else if(gpu){
    printf("\n\n９−３．GPU 非再帰 ビット＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
  }else if(sgpu){
    printf("\n\n９−３．SGPU 非再帰 ビット＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
  }else if(q27){
    printf("\n\nQ２７．CPUR 再帰 ビット + 対象解除 + BackTrack1 ＋ BackTrack2 + 枝刈り\n");
  }
  if(cpu||cpur||q27){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;
    char t[20];
    int min=9;
    int targetN=17;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      st=clock();
      if(q27){ q27_NQueens(i); }
      else{ bit93_NQueens(i); }
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  return 0;
}
