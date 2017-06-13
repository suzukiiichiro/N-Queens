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
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() N16: 1:07
   ４．対称解除法(回転と斜軸）          NQueen04() N16: 1:09
 <>５．枝刈りと最適化                   NQueen05() N16: 0:18
   ６．ビットマップ                     NQueen06() 
   ７．ビットマップ+対称解除法          NQueen07() 
   ８．ビットマップ+クイーンの場所で分岐NQueen08() 
   ９．ビットマップ+枝刈りと最適化      NQueen09() 
   10．ビットマップ+部分解合成法        NQueen10()
   11．マルチスレッド                   NQueen11()

 * ５．枝刈りと最適化
 * 　単純ですのでソースのコメントを見比べて下さい。
 *   単純ではありますが、枝刈りの効果は絶大です。

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
  15:      2279184          285053      0 00:00:03
  16:     14772512         1846955      0 00:00:18
  17:     95815104        11977939      0 00:02:20
 */
#include<stdio.h>
#include<time.h>

#define MAXSIZE 27

int lTotal=1 ; //合計解
int lUnique=0; //ユニーク解
int iSize;     //Ｎ
int colChk [2*MAXSIZE-1]; //縦 配置フラグ　
int diagChk[2*MAXSIZE-1]; //斜め配置フラグ　
int antiChk[2*MAXSIZE-1]; //斜め配置フラグ　
int aBoard[MAXSIZE];  //チェス盤の横一列
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
  int k=neg?0:n-1;
  int incr=(neg?+1:-1);
  for(int j=0;j<n;k+=incr){ scr[j++]=check[k];}
  k=neg?n-1:0;
  for(int j=0;j<n;k-=incr){ check[scr[j++]]=k;}
}
void vMirror(int check[],int n){
  for(int j=0;j<n;j++){ check[j]=(n-1)- check[j];}
}
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
int symmetryOps(){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<iSize;i++){ aTrial[i]=aBoard[i];}
  rotate(aTrial,aScratch,iSize,0);  //時計回りに90度回転
  int k=intncmp(aBoard,aTrial,iSize);
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
  for(int i=0;i<iSize;i++){ aTrial[i]=aBoard[i];}
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
void NQueen(int row){
  int vTemp;
  if(row==iSize-1){
    // 枝刈り antiChk:右斜め上 dianChk:左斜め上
    if ((diagChk[row-aBoard[row]+iSize-1] ||antiChk[row+aBoard[row]])){   
      return; 
    }
    int k=symmetryOps();//対称解除法
    if(k!=0){
      lUnique++;
      lTotal+=k;
    }
  }else{
    int lim=(row!=0)?iSize:(iSize+1)/2;
    for(int col=row;col<lim;col++){
      //未使用の数字（クイーン）と交換する
      vTemp=aBoard[col]; aBoard[col]=aBoard[row]; aBoard[row]=vTemp;
      // 枝刈り antiChk:右斜め上 dianChk:左斜め上
      if(!(diagChk[row-aBoard[row]+iSize-1]||antiChk[row+aBoard[row]])){
        diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1;
        NQueen(row+1);
        diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0;
      }
    }
    vTemp=aBoard[row];
    for (int i=row+1;i<iSize;i++){
      aBoard[i-1]=aBoard[i];
    }
    aBoard[iSize-1]=vTemp;
  }
}
int main(void){
  clock_t st;
  char t[20];
  printf("%s\n"," N:        Total       Unique        dd:hh:mm:ss");
  //for(int i=2;i<=MAXSIZE;i++){
  for(int i=2;i<=17;i++){
    iSize=i;
    lTotal=0;
    lUnique=0;
    for(int j=0;j<iSize;j++){
      aBoard[j]=j;
    }
    st=clock();
    NQueen(0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",iSize,getTotal(),getUnique(),t);
  } 
}

