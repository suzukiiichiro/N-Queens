/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 ４．対称解除法

		コンパイルと実行
		$ make nq4 && ./07_04NQueen

     一つの解には、盤面を９０度、１８０度、２７０度回転、及びそれらの鏡像の合計
     ８個の対称解が存在する。対照的な解を除去し、ユニーク解から解を求める手法。
 
 ■ユニーク解の判定方法
   全探索によって得られたある１つの解が、回転・反転などによる本質的に変わること
 のない変換によって他の解と同型となるものが存在する場合、それを別の解とはしない
 とする解の数え方で得られる解を「ユニーク解」といいます。つまり、ユニーク解とは、
 全解の中から回転・反転などによる変換によって同型になるもの同士をグループ化する
 ことを意味しています。
 
   従って、ユニーク解はその「個数のみ」に着目され、この解はユニーク解であり、こ
 の解はユニーク解ではないという定まった判定方法はありません。ユニーク解であるか
 どうかの判断はユニーク解の個数を数える目的の為だけに各個人が自由に定義すること
 になります。もちろん、どのような定義をしたとしてもユニーク解の個数それ自体は変
 わりません。
 
   さて、Ｎクイーン問題は正方形のボードで形成されるので回転・反転による変換パター
 ンはぜんぶで８通りあります。だからといって「全解数＝ユニーク解数×８」と単純には
 いきません。ひとつのグループの要素数が必ず８個あるとは限らないのです。Ｎ＝５の
 下の例では要素数が２個のものと８個のものがあります。


 Ｎ＝５の全解は１０、ユニーク解は２なのです。
 
 グループ１: ユニーク解１つ目
--- Q-  - Q---
 Q----  ---- Q
-- Q--  -- Q--
---- Q   Q----
- Q---  --- Q-
 
 グループ２: ユニーク解２つ目
---- Q   Q----  -- Q--  -- Q--  --- Q-  - Q---   Q----  ---- Q
-- Q--  -- Q--   Q----  ---- Q  - Q---  --- Q-  --- Q-  - Q---
 Q----  ---- Q  --- Q-  - Q---  ---- Q   Q----  - Q---  --- Q-
--- Q-  - Q---  - Q---  --- Q-  -- Q--  -- Q--  ---- Q   Q----
- Q---  --- Q-  ---- Q   Q----   Q----  ---- Q  -- Q--  -- Q--

 
   それでは、ユニーク解を判定するための定義付けを行いますが、次のように定義する
 ことにします。各行のクイーンが右から何番目にあるかを調べて、最上段の行から下
 の行へ順番に列挙します。そしてそれをＮ桁の数値として見た場合に最小値になるもの
 をユニーク解として数えることにします。尚、このＮ桁の数を以後は「ユニーク判定値」
 と呼ぶことにします。
 
---- Q   0
-- Q--   2
 Q----   4  --->  0 2 4 1 3 (ユニーク判定値)
--- Q-   1
- Q---   3
 
 
   探索によって得られたある１つの解(オリジナル)がユニーク解であるかどうかを判定
 するには「８通りの変換を試み、その中でオリジナルのユニーク判定値が最小であるか
 を調べる」ことになります。しかし結論から先にいえば、ユニーク解とは成り得ないこ
 とが明確なパターンを探索中に切り捨てるある枝刈りを組み込むことにより、３通りの
 変換を試みるだけでユニーク解の判定が可能になります。
  
 
 ■ユニーク解の個数を求める
   先ず最上段の行のクイーンの位置に着目します。その位置が左半分の領域にあればユ
 ニーク解には成り得ません。何故なら左右反転によって得られるパターンのユニーク判
 定値の方が確実に小さくなるからです。また、Ｎが奇数の場合に中央にあった場合はど
 うでしょう。これもユニーク解には成り得ません。何故なら仮に中央にあった場合、そ
 れがユニーク解であるためには少なくとも他の外側の３辺におけるクイーンの位置も中
 央になければならず、それは互いの効き筋にあたるので有り得ません。


 ■ユニーク解から全解への展開
   これまでの考察はユニーク解の個数を求めるためのものでした。全解数を求めるには
 ユニーク解を求めるための枝刈りを取り除いて全探索する必要があります。したがって
 探索時間を犠牲にしてしまうことになります。そこで「ユニーク解の個数から全解数を
 導いてしまおう」という試みが考えられます。これは、左右反転によるパターンの探索
 を省略して最後に結果を２倍するというアイデアの拡張版といえるものです。そしてそ
 れを実現させるには「あるユニーク解が属するグループの要素数はいくつあるのか」と
 いう考察が必要になってきます。
 
   最初に、クイーンが右上角にあるユニーク解を考えます。斜軸で反転したパターンが
 オリジナルと同型になることは有り得ないことと(×２)、右上角のクイーンを他の３つの
 角に写像させることができるので(×４)、このユニーク解が属するグループの要素数は必
 ず８個(＝２×４)になります。
 
   次に、クイーンが右上角以外にある場合は少し複雑になりますが、考察を簡潔にする
 ために次の事柄を確認します。

 TOTAL=(COUNT8 * 8)+(COUNT4 * 4)+(COUNT2 * 2);
  (1)90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
    ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
    ジナルと同型になる。  

    COUNT2 * 2
 
  (2)90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナ
    ルとは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有
    り得る。 

    COUNT4 * 4
 
  (3)(1)に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを
       加えて２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、
       180度回転させて同型になる場合は４個(左右反転×縦横回転)、そして180度回転させても
       オリジナルと異なる場合は８個になります。(左右反転×縦横回転×上下反転)
 
    COUNT8 * 8 

   以上のことから、ひとつひとつのユニーク解が上のどの種類に該当するのかを調べる
 ことにより全解数を計算で導き出すことができます。探索時間を短縮させてくれる枝刈
 りを外す必要がなくなったというわけです。 
 
   UNIQUE  COUNT2     +  COUNT4     +  COUNT8
   TOTAL (COUNT2 * 2)+(COUNT4 * 4)+(COUNT8 * 8)

 　これらを実現すると、前回のNQueen3()よりも実行速度が遅くなります。
 　なぜなら、対称・反転・斜軸を反転するための処理が加わっているからです。
 ですが、今回の処理を行うことによって、さらにNQueen5()では、
 処理スピードが飛躍的に高速化されます。
 そのためにも今回のアルゴリズム実装は必要なのです。

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
11:         2680             341            0.01
12:        14200            1787            0.04
13:        73712            9233            0.27
14:       365596           45752            1.57
15:      2279184          285053           10.09
16:     14772512         1846955         1:08.64
17:     95815104        11977939         8:11.18
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
//
struct HIKISU{
  int Y;
  int I;
};
//
struct STACK {
  struct HIKISU param[MAX];
  int current;
};
 

void NQueen(int row, int si);
void TimeFormat(clock_t utime,char *form);
int symmetryOps(int si);
void rotate(int chk[],int scr[],int n,int neg);
void vMirror(int chk[],int n);
int intncmp(int lt[],int rt[],int n);

// i:col si:size r:row fA:縦 fB:斜め fC:斜め
void NQueen(int y,int si){
  struct STACK stParam;
  for (int m=0;m<si;m++){ 
    stParam.param[m].Y=0;
    stParam.param[m].I=0;
  }
  stParam.current=0;
  while(1){
  start:
  printf("methodstart\n");
  printf("###y:%d\n",y);
  for(int k=0;k<si;k++){
    printf("###i:%d\n",k);
    printf("###fa[k]:%d\n",fA[k]);
    printf("###fB[k]:%d\n",fB[k]);
    printf("###fC[k]:%d\n",fC[k]);
  }
    if(y==si){
    printf("if(y==si){\n");
      int s=symmetryOps(si);//対称解除法
      if(s!=0){ Unique++; Total+=s; } //解を発見
    printf("Total++;\n");
    }else{
    printf("}else{\n");
      //i:col
      for(int i=0;i<si;i++){
        printf("for(int i=0;i<si;i++){\n");
        aB[y]=i;
        printf("aB[y]=i ;\n");
        printf("###i:%d\n",i);
        printf("###y:%d\n",y);
        for(int k=0;k<si;k++){
          printf("###i:%d\n",k);
          printf("###fa[k]:%d\n",fA[k]);
          printf("###fB[k]:%d\n",fB[k]);
          printf("###fC[k]:%d\n",fC[k]);
        } 
        //バックトラック 制約を満たしているときだけ進む
        if(fA[i]==0 && fB[y-i+(si-1)]==0 && fC[y+i]==0){
        printf("if(fA[i]==0&&fB[y-i+(si-1)]==0&&fC[y+i]==0){\n");
          fA[i]=fB[y-aB[y]+si-1]=fC[y+aB[y]]=1;
        printf("fA[i]=fB[y-aB[y]+si-1]=fC[y+aB[y]]=1;\n");
        printf("###before_nqueen\n");
        printf("###i:%d\n",i);
        printf("###y:%d\n",y);
        for(int k=0;k<si;k++){
          printf("###i:%d\n",k);
          printf("###fa[k]:%d\n",fA[k]);
          printf("###fB[k]:%d\n",fB[k]);
          printf("###fC[k]:%d\n",fC[k]);
        }
        // push(&stParam,i,y); 
        if(stParam.current<MAX){
          stParam.param[stParam.current].I=i;
          stParam.param[stParam.current].Y=y;
          (stParam.current)++;
        }
          y=y+1;
          goto start;
          //NQueen(r+1,si); //再帰
          ret:
          //pop(&stParam);
          if(stParam.current>0){
            stParam.current--;
          }
          y=stParam.param[stParam.current].Y;
          i=stParam.param[stParam.current].I;
        printf("###after_nqueen\n");
        printf("###i:%d\n",i);
        printf("###y:%d\n",y);
        for(int k=0;k<si;k++){
          printf("###i:%d\n",k);
          printf("###fa[k]:%d\n",fA[k]);
          printf("###fB[k]:%d\n",fB[k]);
          printf("###fC[k]:%d\n",fC[k]);
        }
          fA[i]=fB[y-aB[y]+si-1]=fC[y+aB[y]]=0;
        printf("fA[i]=fB[y-aB[y]+si-1]=fC[y+aB[y]]=0;\n");
        }
      printf("}#after:if(fA[i]==0&&fB[y-i+(si-1)]==0&&fC[y+i]==0){\n");
      }  
    printf("after:for\n");
    }
  printf("after:else\n");
    if(y==0){
      break;
    }else{
      goto ret;
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
    NQueen(0,i);
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
// si:size
int symmetryOps(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  rotate(aT,aS,si,0);       //時計回りに90度回転
  int k=intncmp(aB,aT,si);
  if(k>0)return 0;
  if(k==0){ nEquiv=1; }else{
    rotate(aT,aS,si,0);     //時計回りに180度回転
    k=intncmp(aB,aT,si);
    if(k>0)return 0;
    if(k==0){ nEquiv=2; }else{
      rotate(aT,aS,si,0);   //時計回りに270度回転
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }
      nEquiv=4;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aB[i];}
  vMirror(aT,si);           //垂直反転
  k=intncmp(aB,aT,si);
  if(k>0){ return 0; }
  if(nEquiv>1){             //-90度回転 対角鏡と同等       
    rotate(aT,aS,si,1);
    k=intncmp(aB,aT,si);
    if(k>0){return 0; }
    if(nEquiv>2){           //-180度回転 水平鏡像と同等
      rotate(aT,aS,si,1);
      k=intncmp(aB,aT,si);
      if(k>0){ return 0; }  //-270度回転 反対角鏡と同等
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
