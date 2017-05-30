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
   ３．バックトラック                   NQueen3()
   ４．対称解除法(回転と斜軸）          NQueen4()
 <>５．枝刈りと最適化                   NQueen5()
   ６．ビットマップ                     NQueen6()
   ７．マルチスレッド                   NQueen7()

 * ５．枝刈りと最適化
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。

 */
#include<stdio.h>
#include<time.h>

#define MAXSIZE 27

long lTotal=1;
long lUnique=0;
int iSize;
int colChk [2*MAXSIZE-1];
int diagChk[2*MAXSIZE-1];
int antiChk[2*MAXSIZE-1];
int aBoard[MAXSIZE];
int aTrial[MAXSIZE];
int aScratch[MAXSIZE];

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
	return lUnique;
}
long getTotal(){ 
  return lTotal;
}
void rotate(int check[],int scr[],int n,int neg){
  int j;
  int k;
  int incr;
  k=neg?0:n-1;
  incr=(neg?+1:-1);
  for(j=0;j<n;k+=incr){ scr[j++]=check[k];}
  k=neg?n-1:0;
  for(j=0;j<n;k-=incr){ check[scr[j++]]=k;}
}
void vMirror(int check[],int n){
  int j;
  for(j=0;j<n;j++){ check[j]=(n-1)- check[j];}
  return;
}
int intncmp(int lt[],int rt[],int n){
  int k=0;
  int rtn=0;
  for(k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
int symmetryOps(){
  int k ,nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(k=0;k<iSize;k++){ aTrial[k]=aBoard[k];}
  rotate(aTrial,aScratch,iSize,0);  //時計回りに90度回転
  k=intncmp(aBoard,aTrial,iSize);
  if(k>0)return 0;
  if(k==0){ nEquiv=1; }else{
    rotate(aTrial,aScratch,iSize,0);//時計回りに180度回転
    k=intncmp(aBoard,aTrial,iSize);
    if(k>0)return 0;
    if(k==0){ nEquiv=2; }else{
      rotate(aTrial,aScratch,iSize,0);//時計回りに270度回転
      k=intncmp(aBoard,aTrial,iSize);
      if(k>0){ return 0; }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(k=0;k<iSize;k++){ aTrial[k]=aBoard[k];}
  vMirror(aTrial,iSize);    //垂直反転
  k=intncmp(aBoard,aTrial,iSize);
  if(k>0){ return 0; }
  if(nEquiv>1){             //-90度回転 対角鏡と同等       
    rotate(aTrial,aScratch,iSize,1);
    k=intncmp(aBoard,aTrial,iSize);
    if(k>0){return 0; }
    if(nEquiv>2){           //-180度回転 水平鏡像と同等
      rotate(aTrial,aScratch,iSize,1);
      k=intncmp(aBoard,aTrial,iSize);
      if(k>0){ return 0; }  //-270度回転 反対角鏡と同等
      rotate(aTrial,aScratch,iSize,1);
      k=intncmp(aBoard,aTrial,iSize);
      if(k>0){ return 0; }
    }
  }
  return nEquiv * 2;
}
void NQueen5(int row){
  int vTemp;
  if(row==iSize-1){
    if ((diagChk[row-aBoard[row]+iSize-1]||antiChk[row+aBoard[row]])){
      return; //枝刈り
    }
    int k=symmetryOps();
    if(k!=0){
      lUnique++;
      lTotal+=k;
    }
  }else{
		if (!(diagChk[row-aBoard[row]+iSize-1]||antiChk[row+aBoard[row]])){
	    diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1;
			NQueen5(row+1);
	    diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0;
		}
		int lim=(row!=0)?iSize:(iSize+1)/2;
    for(int col=row+1;col<lim;col++){
			vTemp=aBoard[col];
			aBoard[col]=aBoard[row];
			aBoard[row]=vTemp;
			if(!(diagChk[row-aBoard[row]+iSize-1]||antiChk[row+aBoard[row]])){
	      diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1;
	  		NQueen5(row+1);
	      diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0;
			}
    }
		vTemp=aBoard[row];
    for (int i=row+1;i<iSize;i++){
			aBoard[i-1]=aBoard[i];
		}
		//aBoard[i-1]=vTemp;
		aBoard[iSize-1]=vTemp;
 }
 /**
  if(row<iSize-1){
		if (!(diagChk[row-aBoard[row]+iSize-1]||antiChk[row+aBoard[row]])){
	    diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1;
			NQueen5(row+1);
	    diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0;
		}
		lim=(row!=0)?iSize:(iSize+1)/2;
    for(k=row+1;k<lim;k++){
			vTemp=aBoard[k];
			aBoard[k]=aBoard[row];
			aBoard[row]=vTemp;
			if(!(diagChk[row-aBoard[row]+iSize-1]||antiChk[row+aBoard[row]])){
	      diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1;
	  		NQueen5(row+1);
	      diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0;
			}
    }
		vTemp=aBoard[row];
		for (k=row+1;k<iSize;k++){
			aBoard[k-1]=aBoard[k];
		}
		aBoard[k-1]=vTemp;
 }else{
   if ((diagChk[row-aBoard[row]+iSize-1]||antiChk[row+aBoard[row]])){
      return;
   }
   k=symmetryOps();
   if(k!=0){
     lUnique++;
     lTotal+=k;
   }
 }
 return; 
 */
}
int main(void){
  clock_t st;
  char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i;
    lTotal=0;
    lUnique=0;
    for(int j=0;j<iSize;j++){
      aBoard[j]=j;
    }
    st=clock();
    NQueen5(0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",iSize,getTotal(),getUnique(),t);
  } 
}

