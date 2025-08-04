/**
 Cで学ぶアルゴリズムとデータ構造
 ステップバイステップでＮ−クイーン問題を最適化
 一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 実行
 $ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC06.c && ./a.out [-c|-r]


６．CPUR 再帰 バックトラック＋ビットマップ
 

bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 -pthread GCC06.c && ./a.out -r
６．CPUR 再帰 バックトラック＋ビットマップ
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.00
12:        14200               0            0.01
13:        73712               0            0.04
14:       365596               0            0.23
15:      2279184               0            1.40
16:     14772512               0            9.37
17:     95815104               0         1:05.71


bash-3.2$ gcc -Wall -W -O3 -g -ftrapv -std=c99 GCC06.c && ./a.out -c
６．CPU 非再帰 バックトラック＋ビットマップ
 N:        Total       Unique        hh:mm:ss.ms
 4:            2               0            0.00
 5:           10               0            0.00
 6:            4               0            0.00
 7:           40               0            0.00
 8:           92               0            0.00
 9:          352               0            0.00
10:          724               0            0.00
11:         2680               0            0.00
12:        14200               0            0.01
13:        73712               0            0.04
14:       365596               0            0.24
15:      2279184               0            1.47
16:     14772512               0            9.75
17:     95815104               0         1:08.46
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#define THREAD_NUM		96
#define MAX 27
#define TESTSIZE 73712
//変数宣言
int TOTAL=0;         //CPU,CPUR
long UNIQUE=0;        //CPU,CPUR
 const int Maps[] = { 8, 4, 2 };
// 構造体
typedef struct
{
  int x;
  int y;
  int z;
  int w;
}proper ;
//#define nsize     19
#define nsize     8 
#define BLKSZ     1 
//#define BLKSZ     64
#define MPSZ      1 
//#define MPSZ      96
int aBoard[4];
int  N,MASK,J,Mid,X0,MASKH;
long d_cnt;
int T_all,T8,T4,T2;/* 1/2部分解による個数*/
int Kline;
int kubun,Pcnt;
	unsigned int d_counter=0; 
	unsigned int counter=0;			
int rBit[1 << 13];
int Bits[1 << 13];
//-------------------------------------------------------------------------
void NQPart(int n, unsigned int number, proper* Q, int mask,
					 unsigned int T_all, proper* d_proper,int pos)
{
     printf("nq\n");
    unsigned int cnt = 0;	/* cnt : 成功数のカウント */
    int   x,s,h,e,bm,J,K,JM,Mid,MB;
    unsigned int  c,bit,rr;
    unsigned int  bit2,uu,ll;
    
    Mid = (n - 1) / 2;
	MB  = Mid * BLKSZ;
	
	x = number;
    s = x;

	rr = d_proper[pos].x;
	uu = d_proper[pos].y;
	ll = d_proper[pos].z;
	J    = d_proper[pos].w;
	K    = J >> 1;
  printf("j:%d\n",J);
    JM   = J & 0xffff;
  printf("jm:%d\n",JM);
    h    = number + K;
    e    = h + MB;
   printf("rr:%d uu:%d ll:%d\n",rr,uu,ll); 
	bm = rr & ~(uu | ll);
	ll = ll << 1;

    if (x == h)  bm &= JM;
    
    for(;;) {

        if (bm) {
            bit = bit2 = -bm & bm; 
              
            x += BLKSZ;
            Q[x].x = bit;
            Q[x].y = bm ^ bit;
            
            rr =  rr ^ bit;
            uu = (uu | bit2) << 1;
            ll = (ll | (bit2<<1)) >> 1;

            bm = rr & ~(uu | (ll>>1));
        } 
		else {
			bit= bit2 = Q[x].x;
			bm = Q[x].y;
            rr =  rr | bit;
            uu = (uu >> 1) ^ bit2;
            ll = (ll << 1) ^ (bit2<<1);
            
		    x -= BLKSZ;
            if (x == e)   cnt++;
			if (x < s){
        d_counter++;
        c=d_counter;
				if (c < T_all){					
					x = s;
					rr = d_proper[c].x;
                	uu = d_proper[c].y;
					ll = d_proper[c].z;
    				J    = d_proper[c].w;
					K    = J >> 1;
				    JM   = J & 0xffff;
				    h    = number + K;
				    e    = h + MB;
				    
					bm = rr & ~(uu | ll);
					ll = ll << 1;
				}
				else{
          d_cnt+=cnt;
					return;
				}
			}
        }
        if (x == h)  bm &= JM;
	}
}
void NQPart2(int n, unsigned int number, proper* Q, int mask, 
					 unsigned int T_all,  proper* d_proper,int pos)
{
    unsigned int cnt = 0;	/* cnt : 成功数のカウント */
    int   x,s,e,bm,N,Mid;
    unsigned int   c,bit,rr;
    unsigned int  bit2,uu,ll;
    
    N   = (n - 1) / 2;
	Mid = 1 << N;
	
	x = number;
    s = x;
//    e    = number + (n-N-1) * BLKSZ;
    e    = number + (n-N-2) * BLKSZ;

	rr = d_proper[pos].x;
	uu = d_proper[pos].y;
	ll = d_proper[pos].z;
	ll = ll << 1;
//	bm        = Mid;
	bm = d_proper[pos].w;

    for(;;) {

        if (bm) {
            bit = bit2 = -bm & bm;

            x += BLKSZ;
            
            Q[x].x = bit;
            Q[x].y = bm ^ bit;
            rr =  rr ^ bit;
            uu = (uu | bit2) << 1;
            ll = (ll | (bit2<<1)) >> 1;

            bm = rr & ~(uu | (ll>>1));
        } 
		else {
			bit= bit2 = Q[x].x;
			bm = Q[x].y;
            rr =  rr | bit;
            uu = (uu >> 1) ^ bit2;
            ll = (ll << 1) ^ (bit2<<1);

			x -= BLKSZ;
			if (x == e)  cnt++;
			if (x < s){
        d_counter++;
        c=d_counter;
				if(c < T_all){

					x = s;
					rr = d_proper[c].x;
                	uu = d_proper[c].y;
					ll = d_proper[c].z;
					ll = ll << 1;
//					bm        = Mid;
					bm = d_proper[c].w;
				}
				else{
          d_cnt+=cnt;
					return;
				}
			}
        }
    }
}
void assign(int Kbn, int n,  unsigned int T_all, proper* d_proper,  int anumber){
  proper Q[T_all];
	int mask = (1 << n) - 1; 
	
    (Kbn == 0)? 
		NQPart(n, anumber, Q, mask, T_all,  d_proper,anumber) :
		NQPart2(n, anumber, Q, mask, T_all, d_proper,anumber);
	
	return;
}
void  HStore(int Kbn,int X,int U,int V,int BM,int *T,proper *proper)
{
//	printf(" %d   (%x %x %x)\n",*T,X,U&MASK,V);

    proper[*T].x = X ^ MASK;
    proper[*T].y = U;
    proper[*T].z = V;
    proper[*T].w = BM;
    //down,left,right,bitmapっぽいが違うのだろうか
    printf("down:%d left:%d right:%d bm:%d",proper[*T].x,proper[*T].y,proper[*T].z,proper[*T].w);   
    (*T)++;
}
//関数宣言 CPU
void TimeFormat(clock_t utime,char *form);
void NQueen(int size,int mask);
//関数宣言 CPUR
//関数宣言 通常版
//
//
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
int RVS(int X,int N)
{
int k,r,w;
    w = X;
r = 0;
for (k=0; k<N; k++)  { r = (r << 1) | (w & 0x01); w = w >> 1; }
return r;
}
int Bitcount(int X)
{
int c,w;
c = 0;
w = X;
while (w) { c++;  w &= w - 1; }
return c;
}
//CPU 非再帰版 ロジックメソッド
void solve_nqueen(int size,int mask, int row,int h_left,int h_down,int h_right){
	unsigned int left[size];
    unsigned int down[size];
	unsigned int right[size];
    unsigned int bitmap[size];
	left[row]=h_left;
	down[row]=h_down;
	right[row]=h_right;
	bitmap[row]=mask&~(left[row]|down[row]|right[row]);
    unsigned int bit;
    unsigned int sizeE=size-1;
    int mark=row;
    //固定していれた行より上はいかない
    while(row>=mark){//row=1 row>=1, row=2 row>=2
      if(bitmap[row]==0){
        --row;
      }else{
        bitmap[row]^=bit=(-bitmap[row]&bitmap[row]); 
        if((bit&mask)!=0){
          if(row==sizeE){
            TOTAL++;
            --row;
          }else{
            int n=row++;
            left[row]=(left[n]|bit)<<1;
            down[row]=down[n]|bit;
            right[row]=(right[n]|bit)>>1;
            bitmap[row]=mask&~(left[row]|down[row]|right[row]);
          }
        }else{
           --row;
        }
      }  
    }
}
void NQueen(int size,int mask){
  register int sizeE=size-1;
  register int bit;
  if(size<=0||size>32){return;}
  bit=0;
  //bitmap[0]=mask;
  //down[0]=left[0]=right[0]=0;
  //偶数、奇数共通
  for(int col=0;col<size/2;col++){//右側半分だけクイーンを置く
    bit=(1<<col);//
    //down[1]=bit;//再帰の場合は down,left,right,bitmapは現在の行だけで良いが
    //eft[1]=bit<<1;//非再帰の場合は全行情報を配列に入れて行の上がり下がりをする
    //right[1]=bit>>1;
    //bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    //solve_nqueen(size,mask,1,left,down,right);
    solve_nqueen(size,mask,1,bit<<1,bit,bit>>1);
  }
  TOTAL*=2;//ミラーなのでTOTALを２倍する
  //奇数の場合はさらに中央にクイーンを置く
  if(size%2==1){
    bit=(1<<(sizeE)/2);
    //down[1]=bit;
    //left[1]=bit<<1;
    //right[1]=bit>>1;
    //bitmap[1]=mask&~(left[1]|down[1]|right[1]);
    //solve_nqueen(size,mask,1,left,down,right);
    solve_nqueen(size,mask,1,bit<<1,bit,bit>>1);
  }  
}
//
//
//1/4のnqueenを設置してるはず
//ここでc2,c4,c8わけているっぽい
//c2,c4,c8はproper2,T2 prper4,T4 proper8,T8に格納して要るっぽい
//
void NQueenR(int nest,int X,int U,int V,int B,int N,proper* proper2,proper* proper4,proper* proper8)
{
  int b,m,K,JM,BM,A,BB,c;
  int XX,UU,VV;

  if ((nest >= Kline) && (J > 0)) {
    K  = (N - nest) * BLKSZ;
    JM = (1 << J) - 1;
    BM = JM & ~(X | (U<<(N-nest)) | (V>>(N-nest)));
    if (BM){
      HStore(0,X,U,V,(K<<16)+JM,&T8,proper8);
      printf("COUNT8 %d\n",nest);
      for(int i=0;i<nest;i++){
        printf("%d\n",aBoard[i]);
      }  
      TOTAL++;
    }
    return;
   } 
   if (nest == N) {
     A = X & MASKH;
     c = rBit[X >> (N+1)];
     if (Bits[A] == Bits[c]) {
       if (X0 > Mid) {   
         return;
       } 

       BM = Mid & ~(X | U | V);
       if (BM) {
         XX =  X | Mid;
         UU = (U | Mid) << 1;
         VV = (V | Mid) >> 1;
         BM = MASK & ~(XX | UU | VV);
         HStore(2,XX,UU,VV,BM,&T2,proper2);  // D free
         printf("COUNT2\n");
         for(int i=0;i<nest;i++){
           printf("%d\n",aBoard[i]);
         }  
         TOTAL++;
        }
        return;
     }

     if (Bits[A] <  Bits[c]) {
       BM = Mid & ~(X | U | V);
       if (BM) { 
         XX =  X | Mid;
         UU = (U | Mid) << 1;
         VV = (V | Mid) >> 1;
         BM = MASK & ~(XX | UU | VV);
         if (A <  B)  {
           HStore(1,XX,UU,VV,BM,&T4,proper4);
           printf("COUNT4\n");
           for(int i=0;i<nest;i++){
             printf("%d\n",aBoard[i]);
           }  
           TOTAL++;
         }
         if (A == B) { 
           HStore(2,XX,UU,VV,BM,&T2,proper2);
           printf("COUNT2\n");
           for(int i=0;i<nest;i++){
             printf("%d\n",aBoard[i]);
           }  
           TOTAL++;
         }  
       }
     }
     return; 
   }

   m  = MASK & ~(X | U | V);
   while(m != 0) {
     b = m & (-m);
     m = m ^ b;

     if (nest == 0) { 
       if (b == Mid){
          continue; 
        }
        X0 = b;
     }
     if (b == Mid) { J = nest;}
     BB = (b <  Mid)?  B | (1 << nest) : B ;
     aBoard[nest]=b;
     printf("BB:%d,b:%d\n",BB,b);
     NQueenR(nest+1, X|b, (U|b)<<1, (V|b)>>1, BB, N,proper2,proper4,proper8);

     if (b == Mid){  
       J = -1;
     }

   }
}
//
//メインメソッド
int main(int argc,char** argv) {
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1,steps=24576;
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
    printf("\n\n６．CPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(cpur){
    printf("\n\n６．CPUR 再帰 バックトラック＋ビットマップ\n");
  }else if(gpu){
    printf("\n\n６．GPU 非再帰 バックトラック＋ビットマップ\n");
  }else if(sgpu){
    printf("\n\n６．SGPU 非再帰 バックトラック＋ビットマップ\n");
  }
  if(cpu||cpur){
    printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
    clock_t st;          //速度計測用
    char t[20];          //hh:mm:ss.msを格納
    int min=4;
    int targetN=4;
    int mask;
    for(int i=min;i<=targetN;i++){
      TOTAL=0;
      UNIQUE=0;
      mask=((1<<i)-1);
      st=clock();
      //
      //再帰
      if(cpur){ 
        proper proper2[TESTSIZE];
        proper proper4[TESTSIZE];
        proper proper8[TESTSIZE];
        int k;
        N    = nsize >> 1;
        Mid  = 1 << N;
        MASK = (1 << nsize) - 1;
        MASKH= Mid - 1;

        Kline= N - 1; 
        //	Kline= N - 2; 
        J    = -1;

        for (k=0; k<(1<<N); k++)  { rBit[k] = RVS(k,N);  Bits[k] = Bitcount(k); }

        T_all = 0;
        T8    = 0;
        //bitmap_R(size,0,0,0,0);
        printf("N:%d mid:%d mask:%d maskh:%d kline:%d J:%d\n",N,Mid,MASK,MASKH,Kline,J);
        NQueenR(0,0,0,0,0,N,proper2,proper4,proper8);
        //printf("aaaaa");
        printf("T2:%d:T4:%d:T8:%d\n",T2,T4,T8);
        /* 
        for (int j=0;j<T2;j++){
          printf("j:%d",j);
          d_cnt=0;
          assign(0, nsize, T2, proper2,j);
           printf("dcnt:%d",d_cnt);
		      // TOTAL += 2 * *d_cnt;
        }
        */
        /*
        for (int j=0;j<T4;j++){
          d_cnt=0;
          assign(1, nsize, d_cnt, T4, proper4, d_counter,j);

		       TOTAL += 4 * *d_cnt;


        }
        for (int j=0;j<T8;j++){
          d_cnt=0;
          assign(2, nsize, d_cnt, T8, proper8, d_counter,j);

		       TOTAL += 8 * *d_cnt;


        }
        */

      }
      //非再帰
      if(cpu){ 
        NQueen(i,mask); 
      }
      //
      TimeFormat(clock()-st,t);
      printf("%2d:%13ld%16ld%s\n",i,TOTAL,UNIQUE,t);
    }
  }
  return 0;
}
