/**
 BITで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)


３．right/left の導入 バックトラック


 コンパイルと実行
 $ gcc -O3 BIT03_N-Queen.c && ./a.out [-c|-r|-g|-s]
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
typedef struct {
	int left, down, right, bitmap, bit;
} rec;

//bool DEBUG=true;  //デバッグモード
bool DEBUG=false;  //デバッグモード
int aBoard[MAX]; 	//版の配列
int COUNT=0;		//カウント用
long TOTAL=0;
long UNIQUE=0;
void output(int size,rec *d);
void outputR(int size);
void NQueen(int size,int row,int left,int down,int right);
void NQueenR(int size,int row,int left,int down,int right);
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
void NQueen(int size,int row,int left,int down,int right){
	int MASK=((1<<size)-1);
	rec d[size];
	rec *p=d;
	p->left=p->down=p->right=p->bit=0;
	p->bitmap=MASK;
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
			  TOTAL++;
      }
		}
		else if (--p<d) return;
	}
}
//再帰
void NQueenR(int size,int row,int left, int down, int right) {
	int bit,bitmap;
	int MASK=((1<<size)-1);
	int sizeE=size-1;
	for(bitmap=~((left<<=1)|down|(right>>=1))&MASK;bitmap;bitmap&=~bit){
    if(DEBUG){ aBoard[row]=bit=-bitmap&bitmap;
    }else{ bit=-bitmap&bitmap; }
		bit=-bitmap&bitmap;
		if(row<sizeE){
			NQueenR(size,row+1,bit|left,bit|down,bit|right);
		}else{
      if(DEBUG){ outputR(size); }
      TOTAL++;
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
    printf("Usage: %s [-c|-g|-r|-s]\n", argv[0]);
    printf("  -c: CPU only\n");
    printf("  -r: CPUR only\n");
    printf("  -g: GPU only\n");
    printf("  -s: SGPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  if(cpu){ printf("\n\n３．CPU バックトラック\n"); }
  else if(cpur){ printf("\n\n３．CPUR バックトラック\n"); }
  else if(gpu){ printf("\n\n３．GPU バックトラック\n"); }
  else if(sgpu){ printf("\n\n３．SGPU バックトラック\n"); }
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
      if(cpur){ NQueenR(i,0,0,0,0); }
      //CPU
      if(cpu){ NQueen(i,0,0,0,0); }
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
