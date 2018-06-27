/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 コンパイル
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_06N-Queen.c -o 06N-Queen

 実行
 $ ./06N-Queen

 * ６．枝刈りと最適化
 *
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。

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
 13:        73712            9233            0.08
 14:       365596           45752            0.43
 15:      2279184          285053            2.86
 16:     14772512         1846955           18.03
 17:     95815104        11977939         2:15.80
 */

#include <stdio.h>
#include <time.h>
#define MAX 24

long TOTAL=0;
long UNIQUE=0;
int aBoard[MAX];
int fA[2*MAX-1];	//縦列にクイーンを一つだけ配置
int fB[2*MAX-1];	//斜め列にクイーンを一つだけ配置
int fC[2*MAX-1];	//斜め列にクイーンを一つだけ配置
int aT[MAX];       //aT:aTrial[]
int aS[MAX];       //aS:aScrath[]

void NQueen(int row,int size);
int symmetryOps(int si);
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);
void TimeFormat(clock_t utime,char *form);

void NQueen(int row,int size){
	int tmp;
//	if(row==size){
	//枝刈り
	if(row==size-1){
		//枝刈り
		if((fB[row-aBoard[row]+size-1]||fC[row+aBoard[row]])){
			return;
		}
		int s=symmetryOps(size);	//対称解除法の導入
		if(s!=0){
			UNIQUE++;
			TOTAL+=s;
		}
	}else{
		// 枝刈り
		int lim=(row!=0) ? size : (size+1)/2;
		for(int i=row;i<lim;i++){
//		for(int i=0;i<size;i++){
//			aBoard[row]=i;
			// 交換
			tmp=aBoard[i];
			aBoard[i]=aBoard[row];
			aBoard[row]=tmp;
			if(!(fB[row-aBoard[row]+size-1]||fC[row+aBoard[row]])){
				fB[row-aBoard[row]+size-1]=fC[row+aBoard[row]]=1;
				NQueen(row+1,size); //再帰
				fB[row-aBoard[row]+size-1]=fC[row+aBoard[row]]=0;
			}
//			if(fA[i]==0&&fB[row-i+(size-1)]==0&&fC[row+i]==0){
//				fA[i]=fB[row-i+(size-1)]=fC[row+i]=1;
//				NQueen(row+1,size);
//				fA[i]=fB[row-i+(size-1)]=fC[row+i]=0;
//			}
		}
		tmp=aBoard[row];
		for(int i=row+1;i<size;i++){
			aBoard[i-1]=aBoard[i];
		}
		aBoard[size-1]=tmp;
	}
}
int main(void){
	clock_t st;
	char t[20];
	int min=4;
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	for(int i=min;i<=MAX;i++){
		TOTAL=0;
		UNIQUE=0;
		for(int j=0;j<i;j++){
			aBoard[j]=j;
		}
		st=clock();
		NQueen(0,i);
		TimeFormat(clock()-st,t);
		printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
	}
	return 0;
}
int symmetryOps(int size){
	int nEquiv;
	// 回転・反転・対称チェックのためにboard配列をコピー
	for(int i=0;i<size;i++){
		aT[i]=aBoard[i];
	}
	rotate(aT,aS,size,0);       //時計回りに90度回転
	int k=intncmp(aBoard,aT,size);
	if(k>0) return 0;
	if(k==0){
		nEquiv=1;
	}else{
		rotate(aT,aS,size,0);     //時計回りに180度回転
		k=intncmp(aBoard,aT,size);
		if(k>0) return 0;
		if(k==0){
			nEquiv=2;
		}else{
			rotate(aT,aS,size,0);   //時計回りに270度回転
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
	vMirror(aT,size);           //垂直反転
	k=intncmp(aBoard,aT,size);
	if(k>0){
		return 0;
	}
	if(nEquiv>1){             //-90度回転 対角鏡と同等
		rotate(aT,aS,size,1);
		k=intncmp(aBoard,aT,size);
		if(k>0){
			return 0;
		}
		if(nEquiv>2){           //-180度回転 水平鏡像と同等
			rotate(aT,aS,size,1);
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return 0;
			}  //-270度回転 反対角鏡と同等
			rotate(aT,aS,size,1);
			k=intncmp(aBoard,aT,size);
			if(k>0){
				return 0;
			}
		}
	}
	return nEquiv*2;
}
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
void vMirror(int chk[],int n){
	for(int j=0;j<n;j++){
		chk[j]=(n-1)-chk[j];
	}
}
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
