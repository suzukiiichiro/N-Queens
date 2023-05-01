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
  //int bv;
  long down;
  long left;
  long right;
  int aBoard[MAX];
  int topSide;
  int leftSide;
  int bottomSide;
  int rightSide;
  //int BOUND2;
  //int TOPBIT;
  //int SIDEMASK;
  //int LASTMASK;
  long COUNT[3];
}Board ;
long ecnt=0;
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

void print(int size, char *c,int aBoard[])
{
  printf("%s\n", c);
  for (int j = 0; j < size; j++)
  {
    dec2bin(size, 1<<(aBoard[j]));
    printf("\n");
  }
  printf("\n");
  getchar();
}


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
//pressを枝刈りする
bool edakari_1(int size,int x,int y,int pa){
  //角に置いて良いのは右上だけ
  //それ以外の角にクイーンを置いてもsymmetryOpsでスキップされる
  if(pa==size-1||pa==0){
    if(x==0 && y==0){
    }else{
      //ecnt++;
      return false; 
    }
  }
  return true;
}
bool edakari_2(int size,int bpa,int pa)
{
//角の場合は、次のpress_aの値は固定になる
//例えば （上）で1行目が右端だった場合、（右）の1行目は固定になる
  if(bpa==0){
    if(pa !=size-1){
      //ecnt++;
      return false;
    }else{
      return true;
    }
  }
  return true;
}
bool edakari_3(int size,int bpa,int pb)
{
//角から2番目にクイーンを置いたら、次の2行目は0固定
//例:上1行目で左から2列目にクイーンを置いたら、左2行目は0固定
//
 if(bpa ==size-2){
  if(pb !=0){
    //ecnt++;
    return false;
  }
 } 
 return true;
}
bool board_placement(int size,int x,int y,Board *lb)
{
  //int bv;
  long down;
  long left;
  long right;

  long TOPBIT=1<<(size-1);
  long SIDEMASK=(TOPBIT|1);
  long LASTMASK=SIDEMASK;
  for(int a=2;a<=lb->aBoard[0];a++){
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
  }
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
      //if((lb->SIDEMASK&1<<y)==1){
      if((SIDEMASK&1<<y)==1){
        return false;
      }
    //}else if(x>lb->BOUND2){
    }else if(x>size-1-lb->aBoard[0]){
  //【枝刈り】下部サイド枝刈り
      //if((lb->SIDEMASK&1<<y)==1){
      if((SIDEMASK&1<<y)==1){
        return false;
      }
    }
  //【枝刈り】 最下段枝刈り
    //printf("ab:%d,LM;%d\n",lb->aBoard[0],lb->LASTMASK);
    if(x==size-1){
      //if((lb->LASTMASK&1<<y)!=0){
      if((LASTMASK&1<<y)!=0){
        return false;
      }
    }
  }
  if(lb->aBoard[x]==y){  //同じ場所に置くかチェック
    return true;    //同じ場所に置くのはOK
  }
  //bv=1<<x;//xは行 yは列 p.N-1-x+yは右上から左下 x+yは左上から右下
  down=1<<y;
  left=1<<(size-1-x+y);
  right=1<<(x+y);
  //if((lb->bv&bv)||(lb->down&down)||(lb->left&left)||(lb->right&right)){
  if(lb->aBoard[x]!=-1){
    return false;
  }
  if((lb->down&down)||(lb->left&left)||(lb->right&right)){
    return false;
  }     
  lb->aBoard[x]=y;
  //lb->bv|=bv;
  lb->down|=down;
  lb->left|=left;
  lb->right|=right;
  return true;
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
//long bit93_countCompletions(int size,int row,int bv,long left,long down,long right,int sym)
long bit93_countCompletions(int size,int row,int aBoard[],long left,long down,long right,int sym)
{
  ecnt++;
  long bitmap=0;
  long bit=0;
  long cnt = 0;
  //既にクイーンを置いている行はスキップする
  //while((bv&1)!=0) {
  while(aBoard[row]!=-1&&row<size) {
    //bv>>=1;   //右に１ビットシフト
    left<<=1; //left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト  
    row++; 
  }
  //bv>>=1;
  if(row==size){ 
    //print(size,"全ての行にクイーンを置いた",aBoard);
    return 1; 
  }
  else{
    bitmap=~(left|down|right);   
    while(bitmap>0){
      //aBoard[row]=bit=(-bitmap&bitmap);
      bit=(-bitmap&bitmap);
      //int abit;
      //if(size==5){
      //  abit=(bit<<2);  
      //}else if(size==6){
      //  abit=(bit<<1);  
      //}else{
      //  abit=(bit>>(size-7));    
      //}
      //if(abit==1){
      //  aBoard[row]=0;
      //}else if(abit==2){
      //  aBoard[row]=1;
      //}else if(abit==4){
      //  aBoard[row]=2;
      //}else if(abit==8){
      //  aBoard[row]=3;
      //}else if(abit==16){
      //  aBoard[row]=4;
      //}
      //print(size,"クイーンを配置",aBoard);
      bitmap=(bitmap^bit);
      cnt+=bit93_countCompletions(size,row+1,aBoard,(left|bit)<<1,down|bit,(right|bit)>>1,sym);
    }
  }
  return cnt;
}
void bit93_NQueens(int size)
{
  int pres_a[930];
  int pres_b[930];
  int idx=0;
  // int sym;
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
  //for(int topSide=0;topSide<=(size/2)*(size-3);topSide++){
  for(int topSide=0;topSide<=(size/2)*(size-3);topSide++){
    int lsize=(size-2)*(size-1)-topSide;
    Board lBoard;
    lBoard.topSide=topSide;
    //Board leftSideB,rightSideB,bottomSideB,topSideB;
    //int TOPBIT=1<<(size-1);
    //int SIDEMASK=(TOPBIT|1);
    //int LASTMASK=SIDEMASK;
    // int BOUND2=size-2;
    //lBoard.TOPBIT=1<<(size-1);
    //lBoard.SIDEMASK=(lBoard.TOPBIT|1);
    //lBoard.LASTMASK=lBoard.SIDEMASK;
    //int beforepres=0;
    //プログレス
    printf("\r(%d/%d)",topSide,((size/2)*(size-3))); printf("\r"); fflush(stdout);
    //プログレス
    //if(pres_a[topSide]>1 && pres_a[topSide]>beforepres){
    //  lBoard.LASTMASK|=lBoard.LASTMASK>>1|lBoard.LASTMASK<<1;
      //lBoard.BOUND2=size-1-pres_a[topSide];
    //  beforepres=pres_a[topSide];
    //}
    //Board topSideB=lBoard;
    //lBoard=topSideB;
    //lBoard.bv=lBoard.down=lBoard.left=lBoard.right=0;
    lBoard.down=lBoard.left=lBoard.right=0;
    // lBoard.SIDEMASK=SIDEMASK;
    // lBoard.LASTMASK=LASTMASK;
    // lBoard.BOUND2=BOUND2;
    //lBoard.BOUND2=size-2;//元に戻す
    for(int j=0;j<size;j++){ lBoard.aBoard[j]=-1; }
      if(edakari_1(size,0,pres_a[topSide],pres_a[topSide])==false){
        continue;
      }
    //printf("上1行目に配置 (行:%d 列:%d)\n",0,pres_a[topSide]);
    board_placement(size,0,pres_a[topSide],&lBoard);
    //print(size,"配置成功",lBoard.aBoard);
    //printf("上2行目に配置 (行:%d 列:%d)\n",1,pres_b[topSide]);
    if(board_placement(size,1,pres_b[topSide],&lBoard)==false){ 
      //print(size,"配置失敗",lBoard.aBoard);
      continue; 
    }else{
      //print(size,"配置成功または配置済み",lBoard.aBoard);
    }
    Board leftSideB=lBoard;
    for(int leftSide=topSide;leftSide<lsize;leftSide++){
      lBoard=leftSideB;
      lBoard.leftSide=leftSide;
      if(edakari_1(size,pres_a[leftSide],size-1,pres_a[leftSide])==false){
        continue;
      }
      if(edakari_3(size,pres_a[topSide],pres_b[leftSide])==false){
        continue;
      }   
      //if(press_edakari(size,pres_a[topSide],pres_a[leftSide])==false){
      //  continue;
      //}
      //printf("左1行目に配置(行:%d 列:%d)\n",pres_a[leftSide],size-1);
      if(board_placement(size,pres_a[leftSide],size-1,&lBoard)==false){ 
        //print(size,"配置失敗",lBoard.aBoard);
        continue; 
      }else{
        //print(size,"配置成功または配置済み",lBoard.aBoard);
      }
      //printf("左2行目に配置(行:%d 列:%d)\n",pres_b[leftSide],size-2);
      if(board_placement(size,pres_b[leftSide],size-2,&lBoard)==false){ 
        //print(size,"配置失敗",lBoard.aBoard);
        continue; 
      }else{
        //print(size,"配置成功または配置済み",lBoard.aBoard);
      }
      Board bottomSideB=lBoard;
      for(int bottomSide=topSide;bottomSide<lsize;bottomSide++){
        lBoard=bottomSideB;  
        lBoard.bottomSide=bottomSide;
        if(edakari_1(size,size-1,size-1-pres_a[bottomSide],pres_a[bottomSide])==false){
          continue;
        }
        if(edakari_3(size,pres_a[leftSide],pres_b[bottomSide])==false){
          continue;
        }   
        //printf("下1行目に配置(行:%d 列:%d)\n",size-1,size-1-pres_a[bottomSide]);
        if(board_placement(size,size-1,size-1-pres_a[bottomSide],&lBoard)==false){ 
          //print(size,"配置失敗",lBoard.aBoard);
          continue; 
        }else{
          //print(size,"配置成功または配置済み",lBoard.aBoard);
        }
        //printf("下2行目に配置(行:%d 列:%d)\n",size-2,size-1-pres_b[bottomSide]);
        if(board_placement(size,size-2,size-1-pres_b[bottomSide],&lBoard)==false){ 
          //print(size,"配置失敗",lBoard.aBoard);
          continue; 
        }else{
          //print(size,"配置成功または配置済み",lBoard.aBoard);
        }
        Board rightSideB=lBoard;
        for(int rightSide=topSide;rightSide<lsize;rightSide++){
          lBoard=rightSideB;
          lBoard.rightSide=rightSide;
          if(edakari_1(size,size-1-pres_a[rightSide],0,pres_a[rightSide])==false){
          continue;
          }
          if(edakari_2(size,pres_a[topSide],pres_a[rightSide])==false){
            continue;
          }   
          if(edakari_3(size,pres_a[topSide],pres_b[rightSide])==false){
            continue;
          }   
          //printf("右1行目に配置(行:%d 列:%d)\n",size-1-pres_a[rightSide],0);
          if(board_placement(size,size-1-pres_a[rightSide],0,&lBoard)==false){ 
            //print(size,"配置失敗",lBoard.aBoard);
            continue; 
          }else{
            //print(size,"配置成功または配置済み",lBoard.aBoard);
          }
          //printf("右2行目に配置(行:%d 列:%d)\n",size-1-pres_b[rightSide],1);
          if(board_placement(size,size-1-pres_b[rightSide],1,&lBoard)==false){ 
            //print(size,"配置失敗",lBoard.aBoard);
            continue; 
          }else{
            //print(size,"配置成功または配置済み",lBoard.aBoard);
          }
          // lBoard.topSide=topSide;
          // lBoard.bottomSide=bottomSide;
          // lBoard.leftSide=leftSide;
          // lBoard.rightSide=rightSide;
          //int sym=bit93_symmetryOps_n27(size,&lBoard);
          lBoard.sym=bit93_symmetryOps_n27(size,&lBoard);
          if(lBoard.sym !=3){
            for(int j=0;j<=2;j++){ lBoard.COUNT[j]=0; }
            //NQueenの処理
            lBoard.row=2;
            // lBoard.sym=sym;
            //lBoard.bv=lBoard.bv>>2;
            lBoard.left=lBoard.left>>4;
            lBoard.down=((((lBoard.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1;
            lBoard.right=(lBoard.right>>4)<<(size-5);
            //１行目のQが角か角以外かで分岐する
            //if(lBoard.aBoard[0]==0){
            //  lBoard.COUNT[lBoard.sym]+=bit93_countCompletions1(size,lBoard.row,lBoard.bv,lBoard.left,lBoard.down,lBoard.right,lBoard.sym);
            //}else{
              //printf("countCompletionsに入ります\n");
              lBoard.COUNT[lBoard.sym]+=bit93_countCompletions(size,lBoard.row,lBoard.aBoard,lBoard.left,lBoard.down,lBoard.right,lBoard.sym);
            //}
            UNIQUE+=lBoard.COUNT[lBoard.sym];
            if(lBoard.sym==0){ TOTAL+=lBoard.COUNT[lBoard.sym]*2; 
              //printf("countCompletions終了:COUNT:%ld\n",lBoard.COUNT[lBoard.sym]*2);
            }
            else if(lBoard.sym==1){ TOTAL+=lBoard.COUNT[lBoard.sym]*4; 
              //printf("countCompletionsを実行:COUNT:%ld\n",lBoard.COUNT[lBoard.sym]*4);
            
            }
            else if(lBoard.sym==2){ TOTAL+=lBoard.COUNT[lBoard.sym]*8; 
              //printf("countCompletionsを実行:COUNT:%ld\n",lBoard.COUNT[lBoard.sym]*8);
            
            }
            //printf("###########\n");
          }else{
            //printf("symmetryOpsで枝刈り\n");
            //printf("###########\n");
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
  /** パラメータの処理 */
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
    int min=4;
    int targetN=17;
    //min=5;
    //targetN=5;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      ecnt=0;
      st=clock();
      if(q27){ bit93_NQueens(i); }
      else{ bit93_NQueens(i); }
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s::%ld\n",i,TOTAL,UNIQUE,t,ecnt);
    }
  }
  return 0;
}
