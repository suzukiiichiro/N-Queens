/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm C03_N-Queen.c -o C03_N-Queen

 実行
 $ ./C03_N-Queen


 ３．バックトラック

 　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
 さない事が明らかな場合は、それ以降のパターン生成を行わない。
 　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
 　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
 は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
 けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
 を降りても解がないと判明した時点で木を引き返すということができます。


 実行結果
  N:        Total       Unique        hh:mm:ss.ms
  2:            0               0            0.00
  3:            0               0            0.00
  4:            2               0            0.00
  5:           10               0            0.00
  6:            4               0            0.00
  7:           40               0            0.00
  8:           92               0            0.00
  9:          352               0            0.00
 10:          724               0            0.00
 11:         2680               0            0.01
 12:        14200               0            0.05
 13:        73712               0            0.30
 14:       365596               0            1.93
 15:      2279184               0           13.50
 16:     14772512               0         1:39.30
 17:     95815104               0        12:29.59
 */

#include <stdio.h>
#include <time.h>
#define MAX 17
//
long TOTAL=0;
long UNIQUE=0;
int aBoard[MAX];
int fA[2*MAX-1];	//縦列にクイーンを一つだけ配置
int fB[2*MAX-1];	//斜め列にクイーンを一つだけ配置
int fC[2*MAX-1];	//斜め列にクイーンを一つだけ配置
//
//main()以外のメソッドはここに一覧表記させます
void NQueen(int row,int size);
void TimeFormat(clock_t utime,char *form);
// ロジックメソッド
void NQueen(int row,int size){
	if(row==size){ //最後までこれたらカウント
		TOTAL++;
	}else{
		for(int i=0;i<size;i++){
			aBoard[row]=i;
      //縦斜右斜左を判定
			if(fA[i]==0&&fB[row-i+(size-1)]==0&&fC[row+i]==0){ 
				fA[i]=fB[row-i+(size-1)]=fC[row+i]=1;
				NQueen(row+1,size); //再帰
				fA[i]=fB[row-i+(size-1)]=fC[row+i]=0;
			}
		}
	}
}
// メインメソッド
int main(void){
	clock_t st;           //速度計測用
	char t[20];           //hh:mm:ss.msを格納
	int min=4;            //Nの最小値（スタートの値）を格納
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	for(int i=min;i<=MAX;i++){
		TOTAL=0; UNIQUE=0;  //初期化
		for(int j=0;j<i;j++){ aBoard[j]=j; } //版を初期化
		st=clock();         //計測開始
		NQueen(0,i);
		TimeFormat(clock()-st,t); //計測終了
		printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t); //出力
	}
	return 0;
}
//hh:mm:ss.ms形式に処理時間を出力
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
	if(dd)
	sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
	else if(hh)
	sprintf(form,"     %2d:%02d:%05.2f",hh,mm,ss);
	else if(mm)
	sprintf(form,"        %2d:%05.2f",mm,ss);
	else
	sprintf(form,"           %5.2f",ss);
}
