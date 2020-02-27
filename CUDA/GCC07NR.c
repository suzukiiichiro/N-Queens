
//

/**
７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               1            0.00
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1787            0.02
13:        73712            9233            0.09
14:       365596           45752            0.49
15:      2279184          285053            3.22
16:     14772512         1846955           22.76
17:     95815104        11977939         2:42.11

７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法
 N:        Total       Unique        hh:mm:ss.ms
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
14:       365596           45752            0.32
15:      2279184          285053            2.11
16:     14772512         1846955           14.96
17:     95815104        11977939         1:47.53
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define MAX 27
//
int down[2*MAX-1]; 	//down:flagA 縦 配置フラグ
int right[2*MAX-1]; //right:flagB 斜め配置フラグ
int left[2*MAX-1]; 	//left:flagC 斜め配置フラグ
long TOTAL=0;
long UNIQUE=0;
int aBoard[MAX];
int aT[MAX];       	//aT:aTrial[]
int aS[MAX];       	//aS:aScrath[]
int COUNT2,COUNT4,COUNT8;
//関数宣言
void rotate_bitmap(int chk[],int af[],int si);
void vMirror_bitmap(int chk[],int af[],int si);
int intncmp(int lt[],int rt[],int n);
void symmetryOps_bitmap(int size);
void TimeFormat(clock_t utime,char *form);
void NQueenR(int size,int mask,int row,int left,int down,int right);
void NQueen(int size,int mask);
void print(int size);
//出力
void print(int size){
	printf("%ld: ",TOTAL);
	for(int j=0;j<size;j++){
		printf("%d ",aBoard[j]);
	}
	printf("\n");
}
void dtob(int score,int si) {
  int bit=1; char c[si];
  for (int i=0;i<si;i++) {
    if (score&bit){ c[i]='1'; }else{ c[i]='0'; }
    bit<<=1;
  }
  for (int i=si-1;i>=0;i--){ putchar(c[i]); }
  printf("\n");
}
//
int rh(int a,int sz){
  int tmp=0;
  for(int i=0;i<=sz;i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
//
void vMirror_bitmap(int bf[],int af[],int si){
  int score ;
  for(int i=0;i<si;i++) {
    score=bf[i];
    af[i]=rh(score,si-1);
  }
}
//
void rotate_bitmap(int bf[],int af[],int si){
  for(int i=0;i<si;i++){
    int t=0;
    for(int j=0;j<si;j++){
      t|=((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i]=t;                        // y[i] の j ビット目にする
  }
}
//
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
//
long getUnique(){
	return COUNT2+COUNT4+COUNT8;
}
//
long getTotal(){
	return COUNT2*2+COUNT4*4+COUNT8*8;
}
//
void symmetryOps_bitmap(int si){
  int nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aT[i]=aBoard[i];}
  rotate_bitmap(aT,aS,si);    //時計回りに90度回転
  int k=intncmp(aBoard,aS,si);
  if(k>0)return;
  if(k==0){ nEquiv=2;}else{
    rotate_bitmap(aS,aT,si);  //時計回りに180度回転
    k=intncmp(aBoard,aT,si);
    if(k>0)return;
    if(k==0){ nEquiv=4;}else{
      rotate_bitmap(aT,aS,si);//時計回りに270度回転
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
      nEquiv=8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(int i=0;i<si;i++){ aS[i]=aBoard[i];}
  vMirror_bitmap(aS,aT,si);   //垂直反転
  k=intncmp(aBoard,aT,si);
  if(k>0){ return; }
  if(nEquiv>2){             //-90度回転 対角鏡と同等
    rotate_bitmap(aT,aS,si);
    k=intncmp(aBoard,aS,si);
    if(k>0){return;}
    if(nEquiv>4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aS,aT,si);
      k=intncmp(aBoard,aT,si);
      if(k>0){ return;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k=intncmp(aBoard,aS,si);
      if(k>0){ return;}
    }
  }
  if(nEquiv==2){COUNT2++;}
  if(nEquiv==4){COUNT4++;}
  if(nEquiv==8){COUNT8++;}
}
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask){
  int aStack[MAX+2];
  register int* pnStack;
  register int row=0;
  register int bit;
  register int bitmap;
  int odd=size&1;
  int sizeE=size-1;
	/* センチネルを設定-スタックの終わりを示します*/
  aStack[0]=-1;
	/**
  注：サイズが奇数の場合、（サイズ＆1）は真。
  サイズが奇数の場合は2xをループする必要があります
	*/
  for(int i=0;i<(1+odd);++i){
		/**
			クリティカルループ
			この部分を最適化する必要はありません。
		*/
    bitmap=0;
    if(0==i){
      /*中央を除くボードの半分を処理します
        カラム。ボードが5 x 5の場合、最初の行は00011になります。
        クイーンを中央の列に配置することについてはまだです。
      */
	    /* ２で割る */
      int half=size>>1;
      /*サイズの半分のビットマップで右端の1を埋めます
        サイズが7の場合、その半分は3です（残りは破棄します）
        ビットマップはバイナリで111に設定されます。
      */
      bitmap=(1<<half)-1;
      pnStack=aStack+1;/* スタックポインタ */
      aBoard[0]=0;
      down[0]=left[0]=right[0]=0;
    }else{
			/*（奇数サイズのボードの）中央の列を処理します。
         中央の列ビットを1に設定してから設定します
         したがって、最初の行（1つの要素）と次の半分を処理しています。
         ボードが5 x 5の場合、最初の行は00100になり、次の行は00011です。
      */
      bitmap=1<<(size>>1);
      row=1; /*すでに 0 */
			/* 最初の行にはクイーンが1つだけあります（中央の列）*/
      aBoard[0]=bitmap;
      down[0]=left[0]=right[0]=0;
      down[1]=bitmap;
      /* 次の行を実行します。半分だけビットを設定します
         「Y軸」で結果を反転します
      */
      right[1]=(bitmap>>1);
      left[1]=(bitmap<<1);
      pnStack=aStack+1; // スタックポインタ
      /* この行は-1つの要素のみで完了 */
      *pnStack++=0;
      /* ビットマップ-1は、単一の1の左側すべて1です */
      bitmap=(bitmap-1)>>1;
    }
    // クリティカルループ
    while(true){
      /*
         bit = bitmap ^（bitmap＆（bitmap -1））;
         最初の（最小のsig） "1"ビットを取得しますが、それは遅くなります。
      */
      /* これは、2の補数アーキテクチャを想定しています */
      bit=-((signed)bitmap) & bitmap;
      if(0==bitmap){
        /* 前を取得スタックからのビットマップ */
        bitmap=*--pnStack;
        /* センチネルがヒットした場合... */
        if(pnStack==aStack){
          break ;
        }
        --row;
        continue;
      }
      /* このビットをオフにして、再試行しないようにします */
      bitmap&=~bit;
      /* 結果を保存 */
      aBoard[row]=bit;
      /* 処理する行がまだあるか？ */
      if(row<sizeE){
        int n=row++;
        down[row]=down[n]|bit;
        right[row]=(right[n]|bit)>>1;
        left[row]=(left[n]|bit)<<1;
        *pnStack++=bitmap;
        /* 同じ女王の位置を考慮することはできません
           列、同じ正の対角線、または同じ負の対角線
           すでにボード上のクイーン。
        */
        bitmap=mask&~(down[row]|right[row]|left[row]);
        continue;
      }else{
        /* 処理する行はもうありません。解決策が見つかりました。
           ボードの位置としてソリューションを印刷するために、
           printtableへの呼び出しをコメントアウトします
           printtable（size、aBoard、TOTAL + 1）; */
        //++TOTAL;
			  symmetryOps_bitmap(size); /* 対称解除法の追加 */
        bitmap=*--pnStack;
        --row;
        continue;
      }
    }
  }
  /* 鏡像をカウントするために、ソリューションを2倍します */
  //TOTAL*=2;
}
//CPUR 再帰版　ロジックメソッド
void NQueenR(int size,int mask,int row,int left,int down,int right){
  int bit;
	int bitmap=mask&~(left|down|right);
	if(row==size){
		if(!bitmap){
			aBoard[row]=bitmap;
			symmetryOps_bitmap(size);
		}
	}else{
		while(bitmap){
      // bit=(-bitmap&bitmap);
      // bitmap=(bitmap^bit);
			bitmap^=aBoard[row]=bit=(-bitmap&bitmap);
			NQueenR(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
		}
	}
}
//メインメソッド
int main(int argc,char** argv){
  bool cpu=false,cpur=false;
  int argstart=2;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else{ cpur=true;}
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g]\n",argv[0]);
    printf("  -c: CPU Without recursion\n");
    printf("  -r: CPUR Recursion\n");
  }
	if(cpu){
    printf("\n\n７．CPU 非再帰 バックトラック＋ビットマップ＋対称解除法\n");
	}else if(cpur){
    printf("\n\n７．CPUR 再帰 バックトラック＋ビットマップ＋対称解除法\n");
	}
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	clock_t st;           //速度計測用
	char t[20];           //hh:mm:ss.msを格納
	int min=4;
	int targetN=17;
  int mask;
	//int targetN=4;
	for(int i=min;i<=targetN;i++){
		//TOTAL=0; UNIQUE=0;
    COUNT2=COUNT4=COUNT8=0;
    mask=(1<<i)-1;
		st=clock();
		if(cpu){
			/** 非再帰は-1で初期化 */
			for(int j=0;j<=targetN;j++){
				aBoard[j]=-1;
			}
			NQueen(i,mask);
		}
		if(cpur){
			/** 再帰は0で初期化 */
			for(int j=0;j<=targetN;j++){
				/** 【注意】初期化が前のステップと異なります */
				//aBoard[j]=0;
				aBoard[j]=j;
			}
			NQueenR(i,mask,0,0,0,0);
		}
		TimeFormat(clock()-st,t);
		printf("%2d:%13ld%16ld%s\n",i,getTotal(),getUnique(),t);
	}
	return 0;
}
//hh:mm:ss.ms形式に処理時間を出力
void TimeFormat(clock_t utime,char* form){
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


