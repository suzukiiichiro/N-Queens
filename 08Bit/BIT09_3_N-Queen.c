/**
 CUDAで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

９−３．CPUR 再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2
 N:        Total       Unique        hh:mm:ss.ms
 5:           10               2            0.00
 6:            4               1            0.00
 7:           40               6            0.00
 8:           92              12            0.00
 9:          352              46            0.00
10:          724              92            0.00
11:         2680             341            0.00
12:        14200            1788            0.01
13:        73712            9237            0.04
14:       365596           45771            0.23
15:      2279184          285095            1.40

 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
//変数宣言
long TOTAL=0;         //CPU,CPUR
long UNIQUE=0;        //CPU,CPUR
typedef unsigned long long uint64;
typedef struct{
  uint64 bv;
  uint64 down;
  uint64 left;
  uint64 right;
  int cnt;
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  int x[MAX];
  int y[MAX];
}Board ;
//
Board B;
Board b[2457600];
long bcnt=0;

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
//
int symmetryOps_n27(int w,int e,int n,int s,int size){
  //int lsize=(size-2)*(size-1)-w;
  //if(n<w || n>=lsize){
  //  return 0;	
  //}
  //if(e<w || e>=lsize){
  //  return 0;
  //}
  //if(s<w || s>=lsize){
  //  return 0;
  //}
  //// Check for minimum if n, e, s = (N-2)*(N-1)-1-w
  int ww=(size-2)*(size-1)-1-w;
  //新設
  int w2=(size-2)*(size-1)-1;
  //if(s==ww){
  if((s==ww)&&(n<(w2-e))){
    //check if flip about the up diagonal is smaller
    //if(n<(size-2)*(size-1)-1-e){
    //if(n<(w2-e)){
    return 0;
    //}
  }
  //if(e==ww){
  if((e==ww)&&(n>(w2-n))){
    //check if flip about the vertical center is smaller
    //if(n>(size-2)*(size-1)-1-n){
    //if(n>(w2-n)){
    return 0;       
    //}
  }
  //if(n==ww){
  if((n==ww)&&(e>(w2-s))){
    //// check if flip about the down diagonal is smaller
    //if(e>(size-2)*(size-1)-1-s){
    //if(e>(w2-s)){
    return 0;
    //}
  }
  if(s==w){
    if((n!=w)||(e!=w)){
      // right rotation is smaller unless  w = n = e = s
      //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
      return 0;
    }
    //w=n=e=sであれば90度回転で同じ可能性
    //この場合はミラーの2
    return 2;
  }
  if((e==w)&&(n>=s)){
    //e==wは180度回転して同じ
    if(n>s){
      //180度回転して同じ時n>=sの時はsmaller?
      return 0;
    }
    //この場合は4
    return 4;
  }
  return 8;   
  }

//
  bool board_placement(int si,int x,int y){
    //同じ場所に置くかチェック
    //printf("i:%d:x:%d:y:%d\n",i,B.x[i],B.y[i]);
    if(B.x[x]==y){
      //printf("Duplicate x:%d:y:%d\n",x,y);
      ////同じ場所に置くのはOK
      return true;  
    }
    B.x[x]=y;
    //xは行 yは列 p.N-1-x+yは右上から左下 x+yは左上から右下
    uint64 bv=1<<x;
    uint64 down=1<<y;
    B.y[x]=B.y[x]+down;
    uint64 left=1<<(si-1-x+y);
    uint64 right=1<<(x+y);
    //printf("check valid x:%d:y:%d:p.N-1-x+y:%d;x+y:%d\n",x,y,si-1-x+y,x+y);
    //printf("check valid pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
    //printf("bvcheck:%d:bhcheck:%d:bucheck:%d:bdcheck:%d\n",B.bv&bv,B.bh&bh,B.bu&bu,B.bd&bd);
    if((B.bv&bv)||(B.down&down)||(B.left&left)||(B.right&right)){
      //printf("valid_false\n");
      return false;
    }     
    //printf("before pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
    B.bv|=bv;
    B.down|=down;
    B.left|=left;
    B.right|=right;
    //printf("after pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
    //printf("valid_true\n");
    return true;
  }
//
//CPU 非再帰版 ロジックメソッド
void NQueen(int size,int mask,int row,uint64 b,uint64 l,uint64 d,uint64 r){
  int sizeE=size-1;
  int n;
  uint64 bitmap[size];
  uint64 bv[size];
  uint64 left[size];
  uint64 down[size];
  uint64 right[size];
  uint64 bit=0;
  bitmap[row]=mask&~(l|d|r);
  bv[row]=b;
  down[row]=d;
  left[row]=l;
  right[row]=r;
  while(row>=2){
    while((bv[row]&1)!=0) {
      n=row++;
      bv[row]=bv[n]>>1;//右に１ビットシフト
      left[row]=left[n]<<1;//left 左に１ビットシフト
      right[row]=right[n]>>1;//right 右に１ビットシフト
      down[row]=down[n];  
      bitmap[row]=mask&~(left[row]|down[row]|right[row]);    
    }
    bv[row+1]=bv[row]>>1;
    if(bitmap[row]==0){
      --row;
    }else{
      bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); 
      if((bit&mask)!=0||row>=sizeE){
	//if((bit)!=0){
	if(row>=sizeE){
	  TOTAL++;
	  --row;
	}else{
	  n=row++;
	  left[row]=(left[n]|bit)<<1;
	  down[row]=down[n]|bit;
	  right[row]=(right[n]|bit)>>1;
	  bitmap[row]=mask&~(left[row]|down[row]|right[row]);
	  //bitmap[row]=~(left[row]|down[row]|right[row]);    
	}
      }else{
	--row;
      }
      }
    }  
  }
//
//CPUR 再帰版 ロジックメソッド
  void backTrack1(int size,uint64 mask, int row,uint64 bv,uint64 left,uint64 down,uint64 right,int cnt,int BOUND1){
    uint64 bitmap=0;
    uint64 bit=0;
    //既にクイーンを置いている行はスキップする
    while((bv&1)!=0) {
      bv>>=1;//右に１ビットシフト
      left<<=1;//left 左に１ビットシフト
      right>>=1;//right 右に１ビットシフト  
      row++; 
    }
    bv>>=1;
    if(row==size){
      //TOTAL++;
      UNIQUE++;       //ユニーク解を加算
      TOTAL+=cnt;       //対称解除で得られた解数を加算
    }else{
      //bitmap=mask&~(left|down|right);//maskつけると10桁目以降数が出なくなるので外した
      bitmap=~(left|down|right);  
      if(row<BOUND1) {
	bitmap&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
      }
      while(bitmap>0){
	bit=(-bitmap&bitmap);
	bitmap=(bitmap^bit);
	backTrack1(size,mask,row+1,bv,(left|bit)<<1,down|bit,(right|bit)>>1,cnt,BOUND1);
      }

    }
  }
//
//CPUR 再帰版 ロジックメソッド
void backTrack2(int size,uint64 mask, int row,uint64 bv,uint64 left,uint64 down,uint64 right,int cnt,int BOUND1,int BOUND2,int SIDEMASK,int LASTMASK){
  uint64 bitmap=0;
  uint64 bit=0;
  //既にクイーンを置いている行はスキップする
  while((bv&1)!=0) {
    bv>>=1;//右に１ビットシフト
    left<<=1;//left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト  
    row++; 
  }
  bv>>=1;
  if(row==size){
    //TOTAL++;
    UNIQUE++;       //ユニーク解を加算
    TOTAL+=cnt;       //対称解除で得られた解数を加算
  }else{
    //bitmap=mask&~(left|down|right);//maskつけると10桁目以降数が出なくなるので外した
    bitmap=~(left|down|right);  
    /***11 【枝刈り】上部サイド枝刈*********************/
    if(row<BOUND1){             	
      bitmap&=~SIDEMASK;
      /***11 【枝刈り】下部サイド枝刈り*********************/
    }else if(row==BOUND2) {     	
      if((down&SIDEMASK)==0){ return; }
      if((down&SIDEMASK)!=SIDEMASK){ 
	bitmap&=SIDEMASK; 

      }
    }
    while(bitmap>0){
      bit=(-bitmap&bitmap);
      bitmap=(bitmap^bit);
      backTrack2(size,mask,row+1,bv,(left|bit)<<1,down|bit,(right|bit)>>1,cnt,BOUND1,BOUND2,SIDEMASK,LASTMASK);
    }
  }
}
//
//CPUR 再帰版 ロジックメソッド
void NQueenR(int size,uint64 mask, int row,uint64 bv,uint64 left,uint64 down,uint64 right,int cnt){
  uint64 bitmap=0;
  uint64 bit=0;
  //既にクイーンを置いている行はスキップする
  while((bv&1)!=0) {
    bv>>=1;//右に１ビットシフト
    left<<=1;//left 左に１ビットシフト
    right>>=1;//right 右に１ビットシフト  
    row++; 
  }
  bv>>=1;
  if(row==size){
    //TOTAL++;
    UNIQUE++;       //ユニーク解を加算
    TOTAL+=cnt;       //対称解除で得られた解数を加算
  }else{
    //bitmap=mask&~(left|down|right);//maskつけると10桁目以降数が出なくなるので外した
    bitmap=~(left|down|right);   
    while(bitmap>0){
      bit=(-bitmap&bitmap);
      bitmap=(bitmap^bit);
      NQueenR(size,mask,row+1,bv,(left|bit)<<1,down|bit,(right|bit)>>1,cnt);
    }

  }
}
//
void prepare(int size){
  //CPUR
  int pres_a[930];
  int pres_b[930];
  int TOPBIT;
  int ENDBIT;
  int LASTMASK;
  int SIDEMASK;
  TOPBIT=1<<(size-1);
  SIDEMASK=LASTMASK=(TOPBIT|1);
  ENDBIT=(TOPBIT>>1);
  int beforepres=0;
  int idx=0;
  for(int a=0;a<size;a++){
    for(int b=0;b<size;b++){
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){
	continue;
      }     
      pres_a[idx]=a;
      pres_b[idx]=b;
      idx++;
    }
  }
  Board wB=B;
  //for(int w=0;w<idx;w++){
  for (int w = 0; w <= (size / 2) * (size - 3); w++){
    if(pres_a[w]>1 && pres_a[w]>beforepres){
      //printf("pres_a %d, before %d B1 %d B2 %d\n",pres_a[w],beforepres,pres_a[w],size-1-pres_a[w]);
      LASTMASK|=LASTMASK>>1|LASTMASK<<1;
      ENDBIT>>=1;
      beforepres=pres_a[w];
    }

    B=wB;
    B.bv=B.down=B.left=B.right=0;
    for(int j=0;j<size;j++){
      B.x[j]=-1;
    }
    board_placement(size,0,pres_a[w]);
    board_placement(size,1,pres_b[w]);
    Board nB=B;
    int lsize=(size-2)*(size-1)-w;
    for(int n=w;n<lsize;n++){
      //for(int n=0;n<idx;n++){
      B=nB;
      if(board_placement(size,pres_a[n],size-1)==false){
	continue;
      }
      if(board_placement(size,pres_b[n],size-2)==false){
	continue;
      }
      Board eB=B;
      for(int e=w;e<lsize;e++){
	//for(int e=0;e<idx;e++){
	B=eB;  
	if(board_placement(size,size-1,size-1-pres_a[e])==false){
	  continue;
	}
	if(board_placement(size,size-2,size-1-pres_b[e])==false){
	  continue;
	}
	Board sB=B;
	for(int s=w;s<lsize;s++){
	  //for(int s=0;s<idx;s++){
	  B=sB;
	  if(board_placement(size,size-1-pres_a[s],0)==false){
	    continue;
	  }
	  if(board_placement(size,size-1-pres_b[s],1)==false){
	    continue;
	  }
	  if(pres_a[w]==0){
	    B.cnt=8;
	    b[bcnt]=B;
	    bcnt++;      
	  }else{
	    int cnt=symmetryOps_n27(w,e,n,s,size);
	    if(cnt !=0){
	      B.TOPBIT=TOPBIT;
	      B.ENDBIT=ENDBIT;
	      B.LASTMASK=LASTMASK;
	      B.SIDEMASK=SIDEMASK; 
	      B.cnt=cnt;
	      b[bcnt]=B;
	      bcnt++;                
	    }
	  }
	}
	} 
      }
      }
    }
    //メインメソッド
    int main(int argc,char** argv) {
      bool cpu=false,cpur=false,gpu=false,sgpu=false;
      int argstart=1;
      //,steps=24576;
      /** パラメータの処理 */
      if(argc>=2&&argv[1][0]=='-'){
	if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
	else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
	else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
	else if(argv[1][1]=='s'||argv[1][1]=='S'){sgpu=true;}
	else
	  cpur=true;
	argstart=2;
      }
      if(argc<argstart){
	printf("Usage: %s [-c|-g|-r|-s]\n",argv[0]);
	printf("  -c: CPU only\n");
	printf("  -r: CPUR only\n");
	printf("  -g: GPU only\n");
	printf("  -s: SGPU only\n");
	printf("Default to 8 queen\n");
      }
      /** 出力と実行 */
      if(cpu){
	printf("\n\n９−３．CPU 非再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
      }else if(cpur){
	printf("\n\n９−３．CPUR 再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
      }else if(gpu){
	printf("\n\n９−３．GPU 非再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
      }else if(sgpu){
	printf("\n\n９−３．SGPU 非再帰 ビットマップ＋対象解除＋q２７枝刈＋BackTrack1＋BackTrack2\n");
      }
      if(cpu||cpur){
	printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
	clock_t st;          //速度計測用
	char t[20];          //hh:mm:ss.msを格納
	int min=5;
	int targetN=15;
	uint64 mask;
	for(int i=min;i<=targetN;i++){
	  TOTAL=0;
	  UNIQUE=0;
	  mask=((1<<i)-1);
	  int size=i;
	  //事前準備 上下左右2行2列にクイーンを配置する
	  prepare(size);
	  //事前準備が終わってから時間を計測する
	  st=clock();
	  for (long bc=0;bc<=bcnt;bc++){
	    B=b[bc];
	    if(cpur){
	      if(B.x[0]==0){
		backTrack1(i,mask,2,B.bv >> 2,
		    B.left>>4,
		    ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
		    (B.right>>4)<<(size-5),8,B.x[1]);  
	      }else{     
		if(size==5){
		  backTrack2(i,mask,2,B.bv >> 2,
		      B.left>>4,
		      ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
		      (B.right>>4)<<(size-5),B.cnt,B.x[0],size-1-B.x[0],B.SIDEMASK>>2,B.LASTMASK>>2);  
		}else if(size==6){
		  backTrack2(i,mask,2,B.bv >> 2,
		      B.left>>4,
		      ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
		      (B.right>>4)<<(size-5),B.cnt,B.x[0],size-1-B.x[0],B.SIDEMASK>>1,B.LASTMASK>>1);  
		}else{
		  backTrack2(i,mask,2,B.bv >> 2,
		      B.left>>4,
		      ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
		      (B.right>>4)<<(size-5),B.cnt,B.x[0],size-1-B.x[0],B.SIDEMASK<<size-7,B.LASTMASK<<size-7);
		}
	      }


	    }else if(cpu){
	      //CPU
	      NQueen(i,mask,2,B.bv >> 2,
		  B.left>>4,
		  ((((B.down>>2)|(~0<<(size-4)))+1)<<(size-5))-1,
		  (B.right>>4)<<(size-5));  
	    }                
	  }
	  //
	  TimeFormat(clock()-st,t);
	  printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
	}
      }

      return 0;
    }
