/**
 BITで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)


４．ミラー


 コンパイルと実行
 $ gcc -O3 BIT04_N-Queen.c && ./a.out [-c|-r|-g|-s]
                    -c:cpu 
                    -r cpu再帰 
**/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
//
#define THREAD_NUM 96
#define MAX 27
//
//変数宣言
//bool DEBUG=true;  //デバッグモード
bool DEBUG=false;  //デバッグモード
int aBoard[MAX]; 	//版の配列
int COUNT=0;		//カウント用
long TOTAL=0;
long UNIQUE=0;
//
typedef struct {
	int left, down, right, bitmap, bit;
} rec;
//
void output(int size,rec *d);
void outputR(int size);
void NQueen(int size);
void NQueenR(int size,int row,int left,int down,int right,int mask);
void TimeFormat(clock_t utime,char *form);
//
void TimeFormat(clock_t utime,char *form)
{
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
//非再帰用出力
void output(int size,rec *d){
	printf("pattern %d\n", ++COUNT);
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			putchar(d[i].bit&1<<j?'Q':'*');
		}
		putchar('\n');
	}
}
//再帰用出力
void outputR(int size){
	printf("pattern %d\n", ++COUNT);
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			putchar(aBoard[i]&1<<j?'Q':'*');
		}
		putchar('\n');
	}
}
// 非再帰
//０３非再帰
void NQueen(int size){
	int MASK=((1<<size)-1);
	rec d[size];
	rec *p=d;
	p->left=p->down=p->right=p->bit=0;
	p->bitmap=(1<< ((size+1)>>1))-1;
	while(1){
		if(p->bitmap){
			p->bit=-p->bitmap&p->bitmap;
			p->bitmap&=~p->bit;
			if (p-d<size-1){
				rec *p0=p++;
				p->left=(p0->left|p0->bit)<<1;
				p->down=p0->down|p0->bit;
				p->right=(p0->right|p0->bit)>>1;
				p->bitmap=~(p->left|p->down|p->right)&MASK;
			}else{
			  if(DEBUG){ output(size,d); }
			  TOTAL+=1 + (!(size & 1)||d->bitmap);
      }
		}
		else if (--p<d) return;
	}
}
//再帰
void NQueenR(int size,int row,int left, int down, int right,int mask) {
	//int bit,bitmap;
	int bit;
	int sizeE=size-1;
	for(int bitmap=~((left<<=1)|down|(right>>=1))&mask;bitmap;bitmap&=~bit){
    if(DEBUG){ aBoard[row]=bit=-bitmap&bitmap;
    }else{ bit=-bitmap&bitmap; }
		if(row<sizeE){
			//２行目以降はmaskを戻す
			NQueenR(size,row+1,bit|left,bit|down,bit|right,((1<<size)-1));
		}else{
      if(DEBUG){ outputR(size); }
      //Nが偶数または,Nが奇数でクイーンが中央に置かれていない場合は２加算する
      TOTAL+=1 + (!(size & 1)||!(aBoard[0]&(1<<(size/2))));
      //TOTAL+=1 + (!(size & 1)||bitmap);
    }
	}
}
//メインメソッド
int main(int argc, char **argv){
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1;
  if (argc>=2&&argv[1][0]=='-'){
    if (argv[1][1]=='c'||argv[1][1]=='C'){ cpu = true; }
    else if(argv[1][1]=='r'||argv[1][1]=='R'){ cpur = true; }
    else if(argv[1][1]=='g'||argv[1][1]=='G'){ gpu = true; }
    else if(argv[1][1]=='s'||argv[1][1]=='S'){ sgpu = true; }
    else{ cpur=true; }
    argstart = 2;
  }
  if(argc < argstart){
    printf("Usage: %s [-c|-r]\n", argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
  }
  /** 出力と実行 */
  if(cpu){ printf("\n\n４．CPU 対称解除法\n"); }
  else if(cpur){ printf("\n\n４．CPUR 対称解除法\n"); }
  else if(gpu){ printf("\n\n４．GPUR 対称解除法\n"); }
  else if(sgpu){ printf("\n\n４．SGPU 対称解除法\n"); }
  if(cpu||cpur){
    printf("%s\n", " N:        Total       Unique        hh:mm:ss.ms");
    clock_t st; //速度計測用
    char t[20]; //hh:mm:ss.msを格納
    int min=4;int targetN=17;
    //int mask;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      st=clock();
      //1行目は右半分だけクイーンを置く
      //奇数の場合は中央にもクイーンを置く
      if(cpur){ NQueenR(i,0,0,0,0,(1<< ((i+1)>>1))-1); }
      //CPU
      if(cpu){ NQueen(i); }
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n", i, TOTAL, UNIQUE, t);
    }
  }
  if(gpu||sgpu){
    int min=5;
    int targetN=17;
    //int min=8;int targetN=8;
    struct timeval t0;
    struct timeval t1;
    int ss;
    int ms;
    int dd;
    printf("%s\n", " N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL); // 計測開始
      if(gpu){
        TOTAL=0;
        UNIQUE=0;
        //NQueenG(i,steps);
      }else if(sgpu){
        printf("準備中");
        //TOTAL=sgpu_solve_nqueen_cuda(i,steps);
      }
      gettimeofday(&t1,NULL); // 計測終了
      if(t1.tv_usec<t0.tv_usec){
        dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else{
        dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss %=60;
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n", i, TOTAL, UNIQUE, dd, hh, mm, ss, ms);
    }
  }
  return 0;
}


