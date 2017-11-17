/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

  ８．ビットマップ＋クイーンの場所で分岐

		コンパイルと実行
		$ make nq8 && ./07_08NQueen

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
 * TOTAL = (C8 * 8) + (C4 * 4) + (C2 * 2);
 *   (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
 *    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
 *    ジナルと同型になる。  
 *
 *    C2 * 2
 * 
 *   (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
 *    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
 *    り得る。 
 *
 *    C4 * 4
 * 
 *   (3) (1) に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
 *       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
 *       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
 *       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 * 
 *    C8 * 8 
 *
 *   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 * ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 * りを外す必要がなくなったというわけです。 
 * 
 *   UNIQUE  C2      +  C4      +  C8
 *   TOTAL  (C2 * 2) + (C4 * 4) + (C8 * 8)
 *
 * 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 * 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 * ですが、今回の処理を行うことによって、さらにNQueen5()では、処理スピードが飛
 * 躍的に高速化されます。そのためにも今回のアルゴリズム実装は必要なのです。
 *

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
12:        14200            1787            0.01
13:        73712            9233            0.05
14:       365596           45752            0.33
15:      2279184          285053            2.03
16:     14772512         1846955           16.78
17:     95815104        11977939         1:51.83
 */

#include<stdio.h>
#include<time.h>
#include <math.h>

#define MAX 8 

long Total=1 ; //合計解
long Unique=0; //ユニーク解
int aB[MAX];  //チェス盤の横一列
int aT[MAX];
int aS[MAX];
int C2=0;int C4=0;int C8=0;

int BOUND1;
int BOUND2;
int TOPBIT;
int ENDBIT;
int SIDEMASK;
int LASTMASK;

void symmetryOps_bm(int si);
int intncmp(int lt[],int rt[],int si);
int rh(int a,int sz);
long getTotal();
long getUnique();
void dtob(int score,int si);
void vMirror_bitmap(int bf[],int af[],int si);
void rotate_bitmap(int bf[],int af[],int si);
void TimeFormat(clock_t utime,char *form);
void backTrack2(int is,int msk,int y, int l, int d, int r);
void backTrack1(int si,int msk,int y, int l, int d, int r);
struct HIKISU{
  int Y;
  int I;
  int M;
  int L;
  int D;
  int R;
  int B;
};
struct STACK {
  struct HIKISU param[MAX];
  int current;
};
void NQueen(int si,int msk);

void backTrack2(int si,int msk,int y,int l,int d,int r){
  struct STACK stParam_2;
  for (int m=0;m<si;m++){ 
    stParam_2.param[m].Y=0;
    stParam_2.param[m].I=si;
    stParam_2.param[m].M=0;
    stParam_2.param[m].L=0;
    stParam_2.param[m].D=0;
    stParam_2.param[m].R=0;
    stParam_2.param[m].B=0;
  }
  stParam_2.current=0;
  int bend_2=0;
  int rflg_2=0;
  int bit;
  int bm;
  while(1){
//start:
    printf("methodstart:backtrack2\n");
    printf("###y:%d\n",y);
    printf("###l:%d\n",l);
    printf("###d:%d\n",d);
    printf("###r:%d\n",r);
    for(int k=0;k<si;k++){
      printf("###i:%d\n",k);
      printf("###aB[k]:%d\n",aB[k]);
    }
    if(rflg_2==0){
      bm=msk&~(l|d|r); /* 配置可能フィールド */
    }
    if (y==si&&rflg_2==0) {
      printf("if(y==si){\n");
      if(!bm){
        aB[y]=bm;
        symmetryOps_bm(si);
      }
    }else{
      printf("}else{#y==si\n");
      while(bm|| rflg_2==1) {
        printf("while(bm){\n");
        if(rflg_2==0){
          bm^=aB[y]=bit=(-bm&bm); //最も下位の１ビットを抽出
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
          if(stParam_2.current<MAX){
            stParam_2.param[stParam_2.current].Y=y;
            stParam_2.param[stParam_2.current].I=si;
            stParam_2.param[stParam_2.current].M=msk;
            stParam_2.param[stParam_2.current].L=l;
            stParam_2.param[stParam_2.current].D=d;
            stParam_2.param[stParam_2.current].R=r;
            stParam_2.param[stParam_2.current].B=bm;
            (stParam_2.current)++;
          }
          y=y+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_2=1;
          break;
        }
        //goto start;
        //backTrack2(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
        //ret:
        if(rflg_2==1){ 
          if(stParam_2.current>0){
            stParam_2.current--;
          }
          si=stParam_2.param[stParam_2.current].I;
          y=stParam_2.param[stParam_2.current].Y;
          msk=stParam_2.param[stParam_2.current].M;
          l=stParam_2.param[stParam_2.current].L;
          d=stParam_2.param[stParam_2.current].D;
          r=stParam_2.param[stParam_2.current].R;
          bm=stParam_2.param[stParam_2.current].B;
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
          rflg_2=0;
        }
      }
      printf("}:end while(bm){\n");
      if(bend_2==1 && rflg_2==0){
        bend_2=0;
        continue;
      }
    } 
    printf("}:end else\n");
    if(y==1){
      break;
    }else{
      printf("gotoreturn\n");
      //goto ret;
      rflg_2=1;
    }
  }
}
void backTrack1(int si,int msk,int y,int l,int d,int r){
  struct STACK stParam_1;
  for (int m=0;m<si;m++){ 
    stParam_1.param[m].Y=0;
    stParam_1.param[m].I=si;
    stParam_1.param[m].M=0;
    stParam_1.param[m].L=0;
    stParam_1.param[m].D=0;
    stParam_1.param[m].R=0;
    stParam_1.param[m].B=0;
  }
  stParam_1.current=0;
  int bend_1=0;
  int rflg_1=0;
  int bit;
  int bm;
  while(1){
//start:
    printf("methodstart:backtrack1\n");
    printf("###y:%d\n",y);
    printf("###l:%d\n",l);
    printf("###d:%d\n",d);
    printf("###r:%d\n",r);
    for(int k=0;k<si;k++){
      printf("###i:%d\n",k);
      printf("###aB[k]:%d\n",aB[k]);
    }
    if(rflg_1==0){
      bm=msk&~(l|d|r); /* 配置可能フィールド */
    }
    if (y==si&&rflg_1==0) {
      printf("if(y==si){\n");
      if(!bm){
        aB[y]=bm;
        symmetryOps_bm(si);
      }
    }else{
      printf("}else{#y==si\n");
      while(bm|| rflg_1==1) {
        printf("while(bm){\n");
        if(rflg_1==0){
          bm^=aB[y]=bit=(-bm&bm); //最も下位の１ビットを抽出
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
          if(stParam_1.current<MAX){
            stParam_1.param[stParam_1.current].Y=y;
            stParam_1.param[stParam_1.current].I=si;
            stParam_1.param[stParam_1.current].M=msk;
            stParam_1.param[stParam_1.current].L=l;
            stParam_1.param[stParam_1.current].D=d;
            stParam_1.param[stParam_1.current].R=r;
          stParam_1.param[stParam_1.current].B=bm;
            (stParam_1.current)++;
          }
          y=y+1;
          l=(l|bit)<<1;
          d=(d|bit);
          r=(r|bit)>>1;
          bend_1=1;
          break;
        }
        //goto start;
        //backTrack1(si,msk,y+1,(l|bit)<<1,d|bit,(r|bit)>>1);
//ret:
        if(rflg_1==1){ 
        if(stParam_1.current>0){
          stParam_1.current--;
        }
        si=stParam_1.param[stParam_1.current].I;
        y=stParam_1.param[stParam_1.current].Y;
        msk=stParam_1.param[stParam_1.current].M;
        l=stParam_1.param[stParam_1.current].L;
        d=stParam_1.param[stParam_1.current].D;
        r=stParam_1.param[stParam_1.current].R;
        bm=stParam_1.param[stParam_1.current].B;
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
          rflg_1=0;
        }
      }
      printf("}:end while(bm){\n");
      if(bend_1==1 && rflg_1==0){
        bend_1=0;
        continue;
      }
    } 
    printf("}:end else\n");
    if(y==2){
      break;
    }else{
      printf("gotoreturn\n");
      //goto ret;
      rflg_1=1;
    }
  }
}
void NQueen(int si,int msk){
  int bit;
  TOPBIT=1<<(si-1);
  aB[0]=1;
  for(BOUND1=2;BOUND1<si-1;BOUND1++){
    aB[1]=bit=(1<<BOUND1);
    backTrack1(si,msk,2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  for(BOUND1=1,BOUND2=si-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aB[0]=bit=(1<<BOUND1);
    backTrack2(si,msk,1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
int main(void){
  clock_t st; char t[20];
  int min=8;int msk;
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
  int score;
  for(int i=0;i< si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
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
  int aT[si];
  int aS[si];
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  rotate_bitmap(aT,aS,si);  //時計回りに90度回転
  int k=intncmp(aB,aS,si);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aS,aT,si);//時計回りに180度回転
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
  vMirror_bitmap(aS,aT,si);    //垂直反転
  k=intncmp(aB,aT,si);
  if(k>0){ return; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等       
    rotate_bitmap(aT,aS,si);
    k=intncmp(aB,aS,si);
    if(k>0){return;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aB,aT,si);
      if(k>0){ return;}  //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aB,aS,si);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){ C2++; }
  if(nEquiv==4){ C4++; }
  if(nEquiv==8){ C8++; }
}
