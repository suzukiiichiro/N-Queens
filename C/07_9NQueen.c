/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  Java版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen
  Bash版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Bash
  Lua版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Lua
 
  ステップバイステップでＮ−クイーン問題を最適化
   １．ブルートフォース（力まかせ探索） NQueen1()
   ２．配置フラグ（制約テスト高速化）   NQueen2()
   ３．バックトラック                   NQueen3() N16: 1:07
   ４．対称解除法(回転と斜軸）          NQueen4() N16: 1:09
   ５．枝刈りと最適化                   NQueen5() N16: 0:18
   ６．ビットマップ                     NQueen6() N16: 0:13
   ７．ビットマップ+対称解除法          NQueen7() N16: 0:21
   ８．ビットマップ+クイーンの場所で分岐NQueen8() N16: 0:13
 <>９．ビットマップ+枝刈りと最適化      NQueen9() N16: 0:02
   10．完成型                           NQueen10() N16: 0:02
   11．マルチスレッド                   NQueen11()

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
15:      2279184          285053      0 00:00:00
16:     14772512         1846955      0 00:00:02
17:     95815104        11977939      0 00:00:15
18:    666090624        83263591      0 00:01:49
19:   4968057848       621012754      0 00:13:53
 */

#include<stdio.h>
#include<time.h>
#include <math.h>
#define MAXSIZE 27

long lTotal=1 ; //合計解
long lUnique=0; //ユニーク解
long COUNT2=0; long COUNT4=0; long COUNT8=0;
int iSize;     //Ｎ
int aBoard[MAXSIZE];  //チェス盤の横一列
int aTrial[MAXSIZE];
int aScratch[MAXSIZE];
int MASK;
int bit;
int BOUND1;
int BOUND2;
int TOPBIT;
int SIZEE;
int SIDEMASK;
int LASTMASK;
int ENDBIT;

/** 時刻のフォーマット変換 */
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
    sprintf(form,"%7d %02d:%02d:%02.0f",dd,hh,mm,ss);
}
/** ユニーク解のget */
long getUnique(){ 
  return COUNT2+COUNT4+COUNT8;
}
/** 総合計のget */
long getTotal(){ 
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
/** 回転 */
void rotate_bitmap(int abefore[],int aafter[]){
  for(int i=0;i<iSize;i++){
    int t=0;
    for(int j=0;j<iSize;j++){
			t|=((abefore[j]>>i)&1)<<(iSize-j-1); // x[j] の i ビット目を
		}
    aafter[i]=t;                        // y[i] の j ビット目にする
  }
}
int rh(int a,int sz){
	int tmp=0;
	for(int i=0;i<=sz;i++){
		if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
	}
	return tmp;
}
/** 鏡像 */
void vMirror_bitmap(int abefore[],int aafter[]){
  for(int i=0;i<iSize;i++) {
    int score=abefore[i];
    aafter[i]=rh(score,iSize-1);
  }
}
int intncmp(int lt[],int rt[]){
  int rtn=0;
  for(int k=0;k<iSize;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
/**********************************************/
/** 対称解除法                               **/
/** ユニーク解から全解への展開               **/
/**********************************************/
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
void symmetryOps_bitmap(){
		if(aBoard[BOUND2]==1){    //90度回転
			int own=1;
			for(int ptn=2;own<=SIZEE;own++,ptn<<=1){
				bit=1;
				for (int you=SIZEE;(aBoard[you]!=ptn)&&(aBoard[own]>=bit);you--){ bit<<=1; }
				if(aBoard[own]>bit){ return; }
				if(aBoard[own]<bit){ break; }
			}
			/** 90度回転して同型なら180度/270度回転も同型である */
			if(own>SIZEE){ COUNT2++; return; }
		}
		if(aBoard[SIZEE]==ENDBIT){ //180度回転
			int own=1;
			for(int you=SIZEE-1;own<=SIZEE;own++,you--){
				bit=1;
				for(int ptn=TOPBIT;(aBoard[you]!=ptn)&&(aBoard[own]>=bit);ptn>>=1){ bit<<=1; }
				if(aBoard[own]>bit){ return; }
				if(aBoard[own]<bit){ break; }
			}
			/** 90度回転が同型でなくても180度回転が同型である事もある */
			if(own>SIZEE){ COUNT4++; return; }
		}
		if(aBoard[BOUND1]==TOPBIT){ //270度回転
			int own=1;
			for(int ptn=TOPBIT>>1;own<=SIZEE;own++,ptn>>=1){
				bit=1;
				for(int you=0;(aBoard[you]!=ptn)&&(aBoard[own]>=bit);you++){ bit<<=1; }
				if(aBoard[own]>bit){ return; }
				if(aBoard[own]<bit){ break; }
			}
		}
		COUNT8++;
}
/**********************************************/
/** 対称解除法                               **/
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
void symmetryOps_bitmap_old(){
  int nEquiv;
  int aTrial[iSize];
  int aScratch[iSize];
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aTrial[i]=aBoard[i];}
  rotate_bitmap(aTrial,aScratch);  //時計回りに90度回転
  int k=intncmp(aBoard,aScratch);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aScratch,aTrial);//時計回りに180度回転
    k=intncmp(aBoard,aTrial);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aTrial,aScratch);//時計回りに270度回転
      k=intncmp(aBoard,aScratch);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aScratch[i]=aBoard[i];}
  vMirror_bitmap(aScratch,aTrial);    //垂直反転
  k=intncmp(aBoard,aTrial);
  if(k>0){ return; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等       
    rotate_bitmap(aTrial,aScratch);
    k=intncmp(aBoard,aScratch);
    if(k>0){return;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aScratch,aTrial);
      k=intncmp(aBoard,aTrial);
      if(k>0){ return;}     //-270度回転 反対角鏡と同等
      rotate_bitmap(aTrial,aScratch);
      k=intncmp(aBoard,aScratch);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){ COUNT2++; }
  if(nEquiv==4){ COUNT4++; }
  if(nEquiv==8){ COUNT8++; }
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
void backTrack2(int y,int left,int down,int right){
  int bitmap=MASK&~(left|down|right); 
  if(y==SIZEE){
    if(bitmap>0){
      if(!(bitmap&LASTMASK)){ //【枝刈り】最下段枝刈り
        aBoard[y]=bitmap;
        symmetryOps_bitmap(); // takakenの移植版の移植版
        //symmetryOps_bitmap_old();// 兄が作成した労作
      }
    }
  }else{
    if(y<BOUND1){             //【枝刈り】上部サイド枝刈り
      bitmap&=~SIDEMASK; 
      // bitmap|=SIDEMASK; 
      // bitmap^=SIDEMASK;(bitmap&=~SIDEMASKと同等)
    }else if(y==BOUND2) {     //【枝刈り】下部サイド枝刈り
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ bitmap&=SIDEMASK; }
    }
    while(bitmap>0) {
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
/**********************************************/
/* 最上段行のクイーンが角にある場合の探索     */
/**********************************************/
/* 
１行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
１行目角にクイーンがある場合、他の角にクイーンを配置することは不可
鏡像についても、主対角線鏡像のみを判定すればよい
２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい 
*/
void backTrack1(int y,int left,int down,int right){
  int bitmap=MASK&~(left|down|right); 
  if(y==SIZEE) {
    if(bitmap>0){
      aBoard[y]=bitmap;
      //【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      COUNT8++;
    }
  }else{
    if(y<BOUND1) {   
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bitmap&=~2; // bitmap|=2; bitmap^=2; (bitmap&=~2と同等)
    }
    while(bitmap>0) {
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  } 
}
/**********************************************/
/** 枝刈りによる高速化 
             最上段の行のクイーンの位置に着目 */
/**********************************************/
/**
ユニーク解に対する左右対称解を予め削除するには、1行目のループのところで、
右半分だけにクイーンを配置するようにすればよい
Nが奇数の場合、クイーンを1行目中央に配置する解は無い。
他の3辺のクィーンが中央に無い場合、その辺が上辺に来るよう回転し、場合に
より左右反転することで、 最小値解とすることが可能だから、中央に配置した
ものしかユニーク解には成り得ないしかし、上辺とその他の辺の中央にクィーン
は互いの効きになるので、配置することが出来ない
*/
void NQueen(int y, int left, int down, int right){
  MASK=(1<<iSize)-1;
  SIZEE=iSize-1;
	TOPBIT=1<<SIZEE;

  /* 最上段行のクイーンが角にある場合の探索 */
  //for(BOUND1=2;BOUND1<SIZEE;BOUND1++){
  for(BOUND1=2;BOUND1<SIZEE-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1); // 角にクイーンを配置 
    backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1)); //２行目から探索
  }

  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);

  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  for(BOUND1=1,BOUND2=SIZEE-1;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2(1,bit<<1,bit,bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
  }
}
int main(void){
  clock_t st; char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i; lTotal=0; lUnique=0;
	  COUNT2=COUNT4=COUNT8=0;
    for(int j=0;j<iSize;j++){ aBoard[j]=j; }
    st=clock();
    NQueen(0,0,0,0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",iSize,getTotal(),getUnique(),t);
  } 
}

