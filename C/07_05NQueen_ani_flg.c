/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
 * ５．枝刈りと最適化

		コンパイルと実行
		$ make nq5 && ./07_05NQueen

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
#include<stdio.h>
#include<time.h>

#define MAX 8 

long Total=1;      //合計解
long Unique=0;      //ユニーク解
int fA [2*MAX-1];   //fA:flagA[] 縦 配置フラグ　
int fB[2*MAX-1];    //fB:flagB[] 斜め配置フラグ　
int fC[2*MAX-1];    //fC:flagC[] 斜め配置フラグ　
int aB[MAX];        //aB:aBoard[] チェス盤の横一列
int aT[MAX];        //aT:aTrial[]
int aS[MAX];        //aS:aScrath[]

void NQueen(int si,int row);
void TimeFormat(clock_t utime,char *form);
int symmetryOps(int si);
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);

struct HIKISU{
  int Y;
  int I;
};
struct STACK {
  struct HIKISU param[MAX];
  int current;
};
// i:col si:size r:row fA:縦 fB:斜め fC:斜め
void NQueen(int si,int r){
  struct STACK stParam;
  for (int m=0;m<si;m++){ 
    stParam.param[m].I=0;
    stParam.param[m].Y=0;
  }
  stParam.current=0;
  int t; //t:temp
  int bend=0;
  int rflg=0;
  while(1){
    //start:
    printf("methodstart\n");
    printf("###r:%d\n",r);
    for(int k=0;k<si;k++){
      printf("###i:%d\n",k);
      printf("###fa[k]:%d\n",fA[k]);
      printf("###fB[k]:%d\n",fB[k]);
      printf("###fC[k]:%d\n",fC[k]);
    }
    if(r==si-1 && rflg==0){
      printf("if(r==si-1){\n");
      // 枝刈り 
      if ((fB[r-aB[r]+si-1]||fC[r+aB[r]])){ 

        //return; 
        printf("skip\n");
        rflg=1;
        //goto ret;
      }
      if(rflg==0){
      int s=symmetryOps(si);//対称解除法
      if(s!=0){ Unique++; Total+=s; } //解を発見
      printf("Total++;\n");
      }
    }else{
      printf("}else{\n");
      // 枝刈り 半分だけ捜査
        int lim=(r!=0)?si:(si+1)/2; 
        printf("lim:%d\n",lim);
      // i:col
      for(int i=r;i<lim;i++){
        printf("for(int i=r;i<lim;i++){\n");
        if(rflg==0){
          t=aB[i]; aB[i]=aB[r]; aB[r]=t; // swap
          printf("t=aB[i]; aB[i]=aB[r]; aB[r]=t;\n");
          printf("###i:%d\n",i);
          printf("###r:%d\n",r);
          for(int k=0;k<si;k++){
            printf("###i:%d\n",k);
            printf("###fa[k]:%d\n",fA[k]);
            printf("###fB[k]:%d\n",fB[k]);
            printf("###fC[k]:%d\n",fC[k]);
          } 
        }
        // 枝刈り バックトラック 制約を満たしているときだけ進む
        if(!(fB[r-aB[r]+si-1]||fC[r+aB[r]])||rflg==1){
          printf("if(!(fB[r-aB[r]+si-1]||fC[r+aB[r]])){\n");
          if(rflg==0){
          fB[r-aB[r]+si-1]=fC[r+aB[r]]=1;
          printf("fB[r-aB[r]+si-1]=fC[r+aB[r]]=1;\n");
          if(stParam.current<MAX){
            stParam.param[stParam.current].I=i;
            stParam.param[stParam.current].Y=r;
            (stParam.current)++;
          }
          r=r+1;
          bend=1;
          break;
          }
          //goto start;
          //NQueen(si,r+1); //再帰
          //ret:
          if(rflg==1){ 
            if(stParam.current>0){
              stParam.current--;
            }
            i=stParam.param[stParam.current].I;
            r=stParam.param[stParam.current].Y;
            printf("###after_nqueen\n");
            printf("###i:%d\n",i);
            printf("###r:%d\n",r);
            for(int k=0;k<si;k++){
              printf("###i:%d\n",k);
              printf("###fa[k]:%d\n",fA[k]);
              printf("###fB[k]:%d\n",fB[k]);
              printf("###fC[k]:%d\n",fC[k]);
              //printf("stParam.parm.si:%d\n",stParam.param[k].SI);
              //printf("stParam.parm.Y:%d\n",stParam.param[k].Y);
            }
            fB[r-aB[r]+si-1]=fC[r+aB[r]]=0;
            printf("fB[r-aB[r]+si-1]=fC[r+aB[r]]=0;\n");
            rflg=0;
          }
        }else{
          bend=0;
        }
      }
      printf("after:for\n");
      if(bend==1 && rflg==0){
        bend=0;
        continue;
      }
      t=aB[r];
      for(int k=r+1;k<si;k++){ aB[k-1]=aB[k]; }
      aB[si-1]=t;
    }
    printf("after:else\n");
    if(r==0){
      break;
    }else{
      printf("gotoreturn\n");
      //goto ret;
      rflg=1;
    }
  }
}
int main(void){
  clock_t st; char t[20];
  int min=8;
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    Total=0; Unique=0;
    for(int j=0;j<i;j++){ aB[j]=j; } //aBを初期化
    st=clock();
    NQueen(i,0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13ld%16ld%s\n",i,Total,Unique,t);
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
//si:size
int symmetryOps(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  rotate(aT,aS,si,0);     //時計回りに90度回転
  int k=intncmp(aB,aT,si);
  if(k>0)return 0;
  if(k==0){ nEquiv=1; }else{
    rotate(aT,aS,si,0);   //時計回りに180度回転
    k=intncmp(aB,aT,si);
    if(k>0)return 0;
    if(k==0){ nEquiv=2; }else{
      rotate(aT,aS,si,0); //時計回りに270度回転
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  vMirror(aT,si);         //垂直反転
  k=intncmp(aB,aT,si);
  if(k>0){ return 0; }
  if(nEquiv>1){           //-90度回転 対角鏡と同等       
    rotate(aT,aS,si,1);
    k=intncmp(aB,aT,si);
    if(k>0){return 0; }
    if(nEquiv>2){         //-180度回転 水平鏡像と同等
      rotate(aT,aS,si,1);
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }//-270度回転 反対角鏡と同等
      rotate(aT,aS,si,1);
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }
    }
  }
  return nEquiv*2;
}
void rotate(int chk[],int scr[],int n,int neg){
  int k=neg?0:n-1;
  int incr=(neg?+1:-1);
  for(int j=0;j<n;k+=incr){ scr[j++]=chk[k];}
  k=neg?n-1:0;
  for(int j=0;j<n;k-=incr){ chk[scr[j++]]=k;}
}
void vMirror(int chk[],int n){
  for(int j=0;j<n;j++){ chk[j]=(n-1)- chk[j];}
}
int intncmp(int lt[],int rt[],int n){
  int rtn=0;
  for(int k=0;k<n;k++){
    rtn=lt[k]-rt[k];
    if(rtn!=0){ break;}
  }
  return rtn;
}
