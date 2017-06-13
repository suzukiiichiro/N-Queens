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
 <>９．ビットマップ+枝刈りと最適化      NQueen9() 
   10．完成型                           NQueen10() N16: 0:02
   11．マルチスレッド                   NQueen11()

  実行結果
 */

#include<stdio.h>
#include<time.h>
#include <math.h>

#define MAXSIZE 27

int lTotal=1 ; //合計解
int lUnique=0; //ユニーク解
int iSize;     //Ｎ
int aBoard[MAXSIZE];  //チェス盤の横一列
int aTrial[MAXSIZE];
int aScratch[MAXSIZE];
int iMask;
int bit;
int COUNT2=0; int COUNT4=0; int COUNT8=0;
int BOUND1;
int BOUND2;
int TOPBIT;
int SIZEE;
int SIDEMASK;
int LASTMASK;
int ENDBIT;


void dtob(int score,int size) {
  //int bit=1,i;
	int bit=1;
  char c[size];
  //for (i=0;i<size;i++) {
  for (int i=0;i<size;i++) {
    if (score&bit){ c[i]='1'; }else{ c[i]='0'; }
    bit<<=1;
  }
  // 計算結果の表示
  for (int i=size-1;i>=0;i--){ putchar(c[i]); }
  printf("\n");
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
    sprintf(form,"%7d %02d:%02d:%02.0f",dd,hh,mm,ss);
}
long getUnique(){ 
  return COUNT2+COUNT4+COUNT8;
}
long getTotal(){ 
  return COUNT2*2+COUNT4*4+COUNT8*8;
}
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
void vMirror_bitmap(int abefore[],int aafter[]){
  for(int i=0;i< iSize;i++) {
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
void symmetryOps_bitmap(int bm){
		//90度回転
		if(aBoard[BOUND2]==1){
			int own=1;
			for(int ptn=2;own<=iSize-1;own++,ptn<<=1){
				bit=1;
				for (int you=iSize-1;(aBoard[you]!=ptn)&&(aBoard[own]>=bit);you--){ bit<<=1; }
				if(aBoard[own]>bit){ return; }
				if(aBoard[own]<bit){ break; }
			}
			/** 90度回転して同型なら180度/270度回転も同型である */
			if (own>iSize-1) { COUNT2++; return; }
		}
		//180度回転
		if(bm==ENDBIT){
			int own=1;
			for(int you=iSize-2;own<=iSize-1;own++,you--){
				bit =1;
				for(int ptn=TOPBIT;(ptn!=aBoard[you])&&(aBoard[own]>=bit);ptn>>=1){ bit<<=1; }
				if(aBoard[own]>bit){ return; }
				if(aBoard[own]<bit){ break; }
			}
			/** 90度回転が同型でなくても180度回転が同型である事もある */
			if(own>iSize-1){
				COUNT4++;
				return;
			}
		}
		//270度回転
		if(aBoard[BOUND1]==TOPBIT){
			int own=1;
			for(int ptn=TOPBIT>>1;own<=iSize-1;own++,ptn>>=1){
				bit=1;
				for(int you=0;aBoard[you]!=ptn&&aBoard[own]>=bit;you++){
					bit<<=1;
				}
				if(aBoard[own]>bit){ return; }
				if (aBoard[own]<bit){ break; }
			}
		}
		COUNT8++;
}
void symmetryOps_bitmap_old(){
  int aTrial[iSize];
  int aScratch[iSize];
  int k;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aTrial[i]=aBoard[i];}
  // 1行目のクイーンのある位置を基準にこれを 90度回転 180度回転 270度回転させた時に重なるかどうか判定する
  // Aの場合 １行目のクイーンのある場所の左端からの列数が1の時（９０度回転したところ）
  if(aBoard[BOUND2]==1){
    rotate_bitmap(aTrial,aScratch);  //時計回りに90度回転
    k=intncmp(aBoard,aScratch);
    if(k>0)return;
    if(k==0){ COUNT2++; return;}
  }
  // Bの場合 1番下の行の現在の左端から2番目が1の場合 180度回転
  if(aBoard[SIZEE]==ENDBIT){
    //aBoard[BOUND2]==1のif 文に入っていない場合は90度回転させてあげる
    if(aBoard[BOUND2]!=1){
      rotate_bitmap(aTrial,aScratch);  //時計回りに90度回転
    }
    rotate_bitmap(aScratch,aTrial);    //時計回りに180度回転
    k=intncmp(aBoard,aTrial);
    if(k>0)return;
    if(k==0){ COUNT4++; return;}
  }
  //Cの場合 1行目のクイーンのある位置の右端からの列数の行の左端が1の時 270度回転
  if(aBoard[BOUND1]==TOPBIT){
    //aBoard[BOUND2]==1のif 文に入っていない場合は90度回転させてあげる
    if(aBoard[BOUND2]!=1){
      rotate_bitmap(aTrial,aScratch);  //時計回りに90度回転
    }
    //aBoard[SIZEE]!=ENDBITのif 文に入っていない場合は180度回転させてあげる
    if(aBoard[SIZEE]!=ENDBIT){
      rotate_bitmap(aScratch,aTrial);//時計回りに180度回転
    }
    rotate_bitmap(aTrial,aScratch);//時計回りに270度回転
    k=intncmp(aBoard,aScratch);
    if(k>0){ return;}
  }
  COUNT8++;
}
void backTrack2(int y,int left,int down,int right){
  int bitmap=iMask&~(left|down|right); 
  if(y==iSize-1){
    if(bitmap){
      /**【枝刈り】最下段枝刈り
       *
       *
       */
      if(!(bitmap&LASTMASK)){
        aBoard[y]=bitmap;
        //symmetryOps_bitmap(bitmap); // takakenの移植版の移植版
        symmetryOps_bitmap_old();// 兄が作成した労作
      }
    }
  }else{
    /**【枝刈り】上部サイド枝刈り
     *
     *
     */
    if(y<BOUND1){                 
      bitmap|=SIDEMASK;
      bitmap^=SIDEMASK;
    /**【枝刈り】下部サイド枝刈り
     *
     *
     */
    }else if(y==BOUND2) {   
      if(!(down&SIDEMASK)){ return; }
      if((down&SIDEMASK)!=SIDEMASK)bitmap&=SIDEMASK;
    }
    while(bitmap) {
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      backTrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
void backTrack1(int y,int left,int down,int right){
  int bitmap=iMask&~(left|down|right); 
  //bitmap は該当行で配置可能 フィールドを1 にする
  // left,down,right は利き筋で配置不可
  /*
  printf("left:%d\n",left);
  dtob(left,iSize-1);
  printf("down:%d\n",down);
  dtob(down,iSize-1);
  printf("right:%d\n",right);
  dtob(right,iSize-1);
  printf("bitmap:%d\n",bitmap);
  dtob(bitmap,iSize-1);
  */
  if(y==iSize-1) {
    if(bitmap){
      aBoard[y]=bitmap;
      /**【枝刈り】斜軸反転解の排除
       *
       */
      COUNT8++;
    }
  }else{
    /**【枝刈り】斜軸反転解の排除
     *
     *
     */
    if(y<BOUND1) {   
      bitmap|=2;
      bitmap^=2;
    }
    while(bitmap) {
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      backTrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  } 
}
void NQueen6(int y, int left, int down, int right){
  SIZEE=iSize-1;
	TOPBIT=1<<SIZEE;

  aBoard[0]=1;
  //printf("aBoard[0]:%d\n",aBoard[0]);
  //dtob(aBoard[0],iSize);
  for(BOUND1=2;BOUND1<iSize-1;BOUND1++){
    aBoard[1]=bit=(1<<BOUND1);
    //printf("aBoard[1]:%d\n",aBoard[1]);
    //dtob(aBoard[1],iSize);
    backTrack1(2,(2|bit)<<1,(1|bit),(bit>>1));
  }
  //SIDEMASK LASTMASK 左端、右端が 1 例: 10000001
  SIDEMASK=LASTMASK=(TOPBIT|1);
  //TOPBIT 左端が 1 例: 1000000
  //ENDBIT 左端から2番目が1 例: 0100000
  ENDBIT=(TOPBIT>>1);
  //printf("LASTMASK:%d\n",LASTMASK);
  //dtob(LASTMASK,iSize);
  //printf("SIDEMASK:%d\n",SIDEMASK);
  //dtob(SIDEMASK,iSize);
  //printf("TOPBIT:%d\n",TOPBIT);
  //dtob(TOPBIT,iSize);
  //printf("ENDBIT:%d\n",ENDBIT);
  //dtob(ENDBIT,iSize);
  for(BOUND1=1,BOUND2=iSize-2;BOUND1<BOUND2;BOUND1++,BOUND2--){
    aBoard[0]=bit=(1<<BOUND1);
    backTrack2(1,bit<<1,bit,bit>>1);
    //両端の1を増やす 1000001 -> 1100011
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    //ENDBITを右にシフトする 01000000 -> 00100000
    ENDBIT>>=1;
    //printf("CHANGELASTMASK:%d\n",LASTMASK);
    //dtob(LASTMASK,iSize);
    //printf("CHANGEENDBIT:%d\n",ENDBIT);
    //dtob(ENDBIT,iSize);
  }
  // 07_7
  // backTrack1(0,0,0,0);
}
int main(void){
  clock_t st; char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i; lTotal=0; lUnique=0;
	  COUNT2=COUNT4=COUNT8=0;
    iMask=(1<<iSize)-1; // 初期化
    for(int j=0;j<iSize;j++){ aBoard[j]=j; }
    st=clock();
    NQueen6(0,0,0,0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",iSize,getTotal(),getUnique(),t);
  } 
}

