// gcc BIT05.c && ./a.out ;

#include <stdio.h>

/**
 * 右半分だけを処理
 * ここからＮを８にします
 */

/**
    N=8 no.1
    - - - - - - - Q 
    - - - Q - - - - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - - Q - - 
    - Q - - - - - - 
    - - - - - - Q - 
    - - - - Q - - - 

    N=8 no.2
    - - - - - - - Q 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - Q - - 
    - Q - - - - - - 
    - - - - Q - - - 
    - - - - - - Q - 
    - - - Q - - - - 

    N=8 no.3
    - - - - - - - Q 
    - Q - - - - - - 
    - - - - Q - - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - - Q - 
    - - - Q - - - - 
    - - - - - Q - - 

    N=8 no.4
    - - - - - - - Q 
    - Q - - - - - - 
    - - - Q - - - - 
    Q - - - - - - - 
    - - - - - - Q - 
    - - - - Q - - - 
    - - Q - - - - - 
    - - - - - Q - - 

    N=8 no.5
    - - - - - - Q - 
    - - - - Q - - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - Q - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - Q - - - - 

    N=8 no.6
    - - - - - - Q - 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - Q - - - 

    N=8 no.7
    - - - - - - Q - 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - Q - - - 
    - - - - - - - Q 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - - Q - - 

    N=8 no.8
    - - - - - - Q - 
    - - Q - - - - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - - Q - - - 
    Q - - - - - - - 
    - - - - - Q - - 
    - - - Q - - - - 

    N=8 no.9
    - - - - - - Q - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - Q - - 
    - - - - - - - Q 
    - - - - Q - - - 
    - Q - - - - - - 
    - - - Q - - - - 

    N=8 no.10
    - - - - - - Q - 
    - Q - - - - - - 
    - - - - - Q - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - Q - - - - 
    - - - - - - - Q 
    - - - - Q - - - 

    N=8 no.11
    - - - - - - Q - 
    - Q - - - - - - 
    - - - Q - - - - 
    Q - - - - - - - 
    - - - - - - - Q 
    - - - - Q - - - 
    - - Q - - - - - 
    - - - - - Q - - 

    N=8 no.12
    - - - - - - Q - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - Q - - - 

    N=8 no.13
    - - - - - Q - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - Q - - - - 
    Q - - - - - - - 
    - - - - - - Q - 
    - - - - Q - - - 
    - - Q - - - - - 

    N=8 no.14
    - - - - - Q - - 
    - - - Q - - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - - Q - - - 
    - - Q - - - - - 

    N=8 no.15
    - - - - - Q - - 
    - - - Q - - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - Q - - - 
    - Q - - - - - - 
    - - - - - - - Q 

    N=8 no.16
    - - - - - Q - - 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - - - - Q 
    - - - - Q - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - Q - - - - - 

    N=8 no.17
    - - - - - Q - - 
    - - - Q - - - - 
    Q - - - - - - - 
    - - - - Q - - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - - - - Q - 
    - - Q - - - - - 

    N=8 no.18
    - - - - - Q - - 
    - - Q - - - - - 
    - - - - - - Q - 
    - - - Q - - - - 
    Q - - - - - - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - - Q - - - 

    N=8 no.19
    - - - - - Q - - 
    - - Q - - - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - - - - - Q 
    - - - - Q - - - 
    Q - - - - - - - 
    - - - Q - - - - 

    N=8 no.20
    - - - - - Q - - 
    - - Q - - - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - Q - - - - 
    - - - - - - - Q 
    Q - - - - - - - 
    - - - - Q - - - 

    N=8 no.21
    - - - - - Q - - 
    - - Q - - - - - 
    - - - - Q - - - 
    - - - - - - - Q 
    Q - - - - - - - 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - - - Q - 

    N=8 no.22
    - - - - - Q - - 
    - - Q - - - - - 
    - - - - Q - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - - - - Q 

    N=8 no.23
    - - - - - Q - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - - - Q 
    - - - - Q - - - 
    - Q - - - - - - 
    - - - Q - - - - 
    - - - - - - Q - 

    N=8 no.24
    - - - - - Q - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - - - Q 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - - - Q - 
    - - - - Q - - - 

    N=8 no.25
    - - - - - Q - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - - Q - 
    - - - - Q - - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - Q - - - - 

    N=8 no.26
    - - - - - Q - - 
    - Q - - - - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - - Q - - - - 
    - - - - - - - Q 
    - - - - Q - - - 
    - - Q - - - - - 

    N=8 no.27
    - - - - - Q - - 
    - Q - - - - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - Q - - - 
    - - - - - - - Q 
    - - - Q - - - - 

    N=8 no.28
    - - - - - Q - - 
    Q - - - - - - - 
    - - - - Q - - - 
    - Q - - - - - - 
    - - - - - - - Q 
    - - Q - - - - - 
    - - - - - - Q - 
    - - - Q - - - - 

    N=8 no.29
    - - - - Q - - - 
    - - - - - - - Q 
    - - - Q - - - - 
    Q - - - - - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - - - Q - - 
    - - Q - - - - - 

    N=8 no.30
    - - - - Q - - - 
    - - - - - - - Q 
    - - - Q - - - - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - - Q - - 
    - Q - - - - - - 
    - - - - - - Q - 

    N=8 no.31
    - - - - Q - - - 
    - - - - - - Q - 
    - - - Q - - - - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    - Q - - - - - - 

    N=8 no.32
    - - - - Q - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - - - Q - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - - - Q 
    - - - Q - - - - 

    N=8 no.33
    - - - - Q - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - - - Q - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - Q - - - - 
    - - - - - - - Q 

    N=8 no.34
    - - - - Q - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - Q - - - - 
    - - - - - - - Q 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - - Q - - 

    N=8 no.35
    - - - - Q - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    - - Q - - - - - 

    N=8 no.36
    - - - - Q - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - Q - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    - - - Q - - - - 
    - Q - - - - - - 

    N=8 no.37
    - - - - Q - - - 
    - - Q - - - - - 
    - - - - - - - Q 
    - - - Q - - - - 
    - - - - - - Q - 
    Q - - - - - - - 
    - - - - - Q - - 
    - Q - - - - - - 

    N=8 no.38
    - - - - Q - - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    - - - Q - - - - 

    N=8 no.39
    - - - - Q - - - 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - Q - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - Q - - - - 
    - - - - - - Q - 

    N=8 no.40
    - - - - Q - - - 
    - Q - - - - - - 
    - - - - - - - Q 
    Q - - - - - - - 
    - - - Q - - - - 
    - - - - - - Q - 
    - - Q - - - - - 
    - - - - - Q - - 

    N=8 no.41
    - - - - Q - - - 
    - Q - - - - - - 
    - - - - - Q - - 
    Q - - - - - - - 
    - - - - - - Q - 
    - - - Q - - - - 
    - - - - - - - Q 
    - - Q - - - - - 

    N=8 no.42
    - - - - Q - - - 
    - Q - - - - - - 
    - - - Q - - - - 
    - - - - - - Q - 
    - - Q - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    Q - - - - - - - 

    N=8 no.43
    - - - - Q - - - 
    - Q - - - - - - 
    - - - Q - - - - 
    - - - - - Q - - 
    - - - - - - - Q 
    - - Q - - - - - 
    Q - - - - - - - 
    - - - - - - Q - 

    N=8 no.44
    - - - - Q - - - 
    Q - - - - - - - 
    - - - - - - - Q 
    - - - - - Q - - 
    - - Q - - - - - 
    - - - - - - Q - 
    - Q - - - - - - 
    - - - Q - - - - 

    N=8 no.45
    - - - - Q - - - 
    Q - - - - - - - 
    - - - - - - - Q 
    - - - Q - - - - 
    - Q - - - - - - 
    - - - - - - Q - 
    - - Q - - - - - 
    - - - - - Q - - 

    N=8 no.46
    - - - - Q - - - 
    Q - - - - - - - 
    - - - Q - - - - 
    - - - - - Q - - 
    - - - - - - - Q 
    - Q - - - - - - 
    - - - - - - Q - 
    - - Q - - - - - 
    count:92
*/
int count;      //見つかった解
int aBoard[8];  //表示用配列
//int *BOARDE;
//int *BOARD1,*BOARD2,
int		TOPBIT,ENDBIT,
		MASK,SIDEMASK,LASTMASK;
//		BOUND1,BOUND2;
int COUNT2,COUNT4,COUNT8;
int SIZE,SIZEE,TOTAL,UNIQUE;
//
// 
//１６進数を２進数に変換
void con(char* c,int decimal){
  int _decimal=decimal;
  int binary=0;
  int base=1;
  while(decimal>0){
    binary=binary+(decimal%2)*base;
    decimal=decimal/2;
    base=base*10;
  }
  printf("%s 16進数:\t%d\t2進数:\t%d\n",c,_decimal,binary);
}
//
//ボード表示用
void Display(void) {
    int  y, bitmap, bit;
    printf("\nN=%d no.%d\n", SIZE, ++count);
    for (y=0; y<SIZE; y++) {
        bitmap = aBoard[y];
        for (bit=1<<(SIZEE); bit; bit>>=1)
            printf("%s ", (bitmap & bit)? "Q": "-");
        printf("\n");
    }
}
/**********************************************/
/* ユニーク解の判定とユニーク解の種類の判定   */
/**********************************************/
void Check(int BOUND1,int BOUND2) {
	int *own,*you,bit,ptn;
	/*90度回転*/
	//if(*BOARD2==1){
	if(aBoard[BOUND2]==1){
		//for(ptn=2,own=aBoard+1;own<=BOARDE;own++,ptn<<=1){
		//for(ptn=2,own=aBoard+1;own<=&aBoard[SIZEE];own++,ptn<<=1){
		ptn=2;
		own=aBoard+1;
		while(own<=&aBoard[SIZEE]){
			bit=1;
			//for(you=BOARDE;*you!=ptn&&*own>=bit;you--)
			//for(you=&aBoard[SIZEE];*you!=ptn&&*own>=bit;you--){
			you=&aBoard[SIZEE];
			while(*you!=ptn&&*own>=bit){
				bit<<=1;
				you--;
			}
			if(*own>bit)return;
			if(*own<bit)break;
			own++;
			ptn<<=1;
		}
		/*90度回転して同型なら180度回転も270度回転も同型である*/
		//if(own>BOARDE){
		if(own>&aBoard[SIZEE]){
			COUNT2++;
			Display(); //表示用
			con("aBoard90",*aBoard);
			return;
		}
	}
	/*180度回転*/
	//if(*BOARDE==ENDBIT){
	if(aBoard[SIZEE]==ENDBIT){
		//for(you=BOARDE-1,own=aBoard+1;own<=BOARDE;own++,you--){
		//for(you=&aBoard[SIZEE]-1,own=aBoard+1;own<=&aBoard[SIZEE];own++,you--){
		you=&aBoard[SIZEE]-1;
		own=aBoard+1;
		while(own<=&aBoard[SIZEE]){
			bit=1;
			//for(ptn=TOPBIT;ptn!=*you&&*own>=bit;ptn>>=1){
			ptn=TOPBIT;
			while(ptn!=*you&&*own>=bit){
				bit<<=1;
				ptn>>=1;
			}
			if(*own>bit)return;
			if(*own<bit)break;
			own++;
			you--;
		}
		/*90度回転が同型でなくても180度回転が同型であることもある*/
		//if(own>BOARDE){
		if(own>&aBoard[SIZEE]){
			COUNT4++;
			Display();  //表示用
			con("aBoard180",*aBoard);
			return;
		}
	}
	/*270度回転*/
	//if(*BOARD1==TOPBIT){
	if(aBoard[BOUND1]==TOPBIT){
		//for(ptn=TOPBIT>>1,own=aBoard+1;own<=BOARDE;own++,ptn>>=1){
		//for(ptn=TOPBIT>>1,own=aBoard+1;own<=&aBoard[SIZEE];own++,ptn>>=1){
		ptn=TOPBIT>>1;
		own=aBoard+1;
		while(own<=&aBoard[SIZEE]){
			bit=1;
			//for(you=aBoard;*you!=ptn&&*own>=bit;you++){
			you=aBoard;
			while(*you!=ptn&&*own>=bit){
				bit<<=1;
				you++;
			}
			if(*own>bit)return;
			if(*own<bit)break;
			own++;
			ptn>>=1;
		}
	}
	COUNT8++;
	Display(); //表示用
	con("aBoard270",*aBoard);
}
/**********************************************/
/* 最上段行のクイーンが角以外にある場合の探索 */
/**********************************************/
void Backtrack2(int y,int left,int down,int right,int BOUND1,int BOUND2){
	int bitmap,bit;

	/**
	left,down,rightだけだと桁数が盤面より多くなる
  （特にleft)ことがあるのでそれを防ぐために使っている 
	*/
	bitmap=MASK&~(left|down|right);
/**
BOUND1 1
BOUND2 6
TOPBIT   10000000
SIDEMASK 10000001 BOUND1が1の時は上部サイド枝狩りはない
LASTMASK 10000001 最終行のクイーンを置けない場所
ENDBIT   01000000
  0 1 2 3 4 5 6 7
0 X - - - - - Q X  aBoard[0]=bit=1<<BOUND1(1) 00000010
1 C - - - - - - -  C:aBoard[BOUND1(1)]==TOPBIT 	aBoard[1]=10000000 270度回転
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - A  A:aBoard[BOUND2(6)]==1 			aBoard[6]=00000001  90度回転
7 X B - - - - - X  B:aBoard[SIZEE(7)]==ENDBIT 	aBoard[7]=01000000 180度回転

BOUND1 2
BOUND2 5
TOPBIT   10000000
SIDEMASK 10000001 y=1,6の時に両サイド枝狩り
LASTMASK 11000011 最終行のクイーンを置けない場所
ENDBIT   00100000 
  0 1 2 3 4 5 6 7
0 X X - - - Q X X  aBoard[0]=bit=1<<BOUND1(2) 00000100
1 X - - - - - - X  
2 C - - - - - - -  C:aBoard[BOUND1(2)]==TOPBIT 	aBoard[2]=10000000 270度回転
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - A  A:aBoard[BOUND2(5)]==1 			aBoard[5]=00000001  90度回転
6 X - - - - - - X  
7 X X B - - - X X  B:aBoard[SIZEE(7)]==ENDBIT 	aBoard[7]=00100000 180度回転

BOUND1 3
BOUND2 4
TOPBIT   10000000
SIDEMASK 10000001 y=1,2,5,6の時に両サイド枝狩り
LASTMASK 11100111 最終行のクイーンを置けない場所
ENDBIT   00010000 
  0 1 2 3 4 5 6 7
0 X X X - Q X X X  aBoard[0]=bit=1<<BOUND1(3) 00001000
1 X - - - - - - X  
2 X - - - - - - X  
3 C - - - - - - -  aBoard[BOUND1]==TOPBIT 			aBoard[3]=10000000 270度回転
4 - - - - - - - A  aBoard[BOUND2]==1 						aBoard[4]=00000001  90度回転
5 X - - - - - - X  
6 X - - - - - - X  
7 X X X B - X X X  aBoard[SIZEE]==ENDBIT 				aBoard[7]=00010000 180度回転
*/
	if(y==SIZEE){
		if(bitmap){
			if(!(bitmap&LASTMASK)){/*最下段枝刈り*/
				aBoard[y]=bitmap;
				//Check();
				Check(BOUND1,BOUND2);
			}
/**
兄　ここの枝刈りの遷移をおねがい
*/
		}
	}else{
		if(y<BOUND1){/*上部サイド枝刈り*/
			bitmap|=SIDEMASK;
			bitmap^=SIDEMASK;
/**
兄　ここの枝刈りの遷移をおねがい
*/
		}else if(y==BOUND2){/*下部サイド枝刈り*/
			if(!(down&SIDEMASK))return;
			if((down&SIDEMASK)!=SIDEMASK)bitmap&=SIDEMASK;
/**
兄　ここの枝刈りの遷移をおねがい
*/
		}
		while(bitmap){
			bitmap^=aBoard[y]=bit=-bitmap&bitmap;
			Backtrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,BOUND2);
		}
	}
}
/**********************************************/
/* 最上段行のクイーンが角にある場合の探索     */
/**********************************************/
void Backtrack1(int y,int left,int down,int right,int BOUND1){
	int  bitmap, bit;
	bitmap=MASK&~(left|down|right);
	if(y==SIZEE){
		if(bitmap){
			aBoard[y]=bitmap;
/**
	兄　ここの変化をお願い
*/
			COUNT8++;
			Display(); //表示用
		}
	}else{
		if(y<BOUND1){/*枝刈り : 斜軸反転解の排除*/
			bitmap|=2;
			bitmap^=2;
/**
	兄　ここの枝刈りをお願い
*/
		}
		while(bitmap){
			bitmap^=aBoard[y]=bit=-bitmap&bitmap;
/**
	兄　ここの枝刈りをお願い
	bitmap^=aBoard[y]=bit=-bitmap&bitmap;
*/
			Backtrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1);
		}
	}
}
//
void NQueens(void) {
	int  bit;
	/*Initialize*/
	COUNT8=COUNT4=COUNT2=0;
	SIZEE=SIZE-1;

	MASK=(1<<SIZE)-1;       //255,11111111

/**
MASK SIZE(8) 	bit
(1<<8)-1:  		255      11111111
  0 1 2 3 4 5 6 7
0 Q Q Q Q Q Q Q Q  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
*/
	/*0行目:000000001(固定)*/
	/*1行目:011111100(選択)*/
	aBoard[0]=1;						//1,  00000001
/**
aBoard[0]=1   1 		00000001
  0 1 2 3 4 5 6 7
0 - - - - - - - Q  
1 - - - - - - - -  
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
*/
	int BOUND1=2;
	//for(BOUND1=2;BOUND1<SIZEE;BOUND1++){
	while(BOUND1<SIZEE){
/**
		aBoard[1]=bit=1<<BOUND1;

BOUND1(2) 	bit
1<<2:  			4      100	
  0 1 2 3 4 5 6 7
0 - - - - - - - Q  aBoard[0]=1
1 X - - - - Q X X  aBoard[1]=bit=1<<BOUND1(2)
2 X - - - - - X X 
3 X - - - - - X X 
4 X - - - - - X X 
5 X - - - - - X X 
6 X - - - - - X X 
7 X - - - - - X X

BOUND1(3)  	bit
1<<3:				8     1000
  0 1 2 3 4 5 6 7
0 - - - - - - - Q  aBoard[0]=1
1 X - - - Q - X X  aBoard[1]=bit=1<<BOUND1(3)
2 X - - - - - X X 
3 X - - - - - X X 
4 X - - - - - X X 
5 X - - - - - X X 
6 X - - - - - X X 
7 X - - - - - X X

BOUND1(4) bit
1<<4:    	16    10000
  0 1 2 3 4 5 6 7
0 - - - - - - - Q  aBoard[0]=1
1 X - - Q - - X X  aBoard[1]=bit=1<<BOUND1(4)
2 X - - - - - X X 
3 X - - - - - X X 
4 X - - - - - X X 
5 X - - - - - X X 
6 X - - - - - X X 
7 X - - - - - X X

BOUND1(5) bit
1<<5:    	32   100000
  0 1 2 3 4 5 6 7
0 X - - - - - - Q  aBoard[0]=1
1 X - Q - - - X X  aBoard[1]=bit=1<<BOUND1(5)
2 X - - - - - X X 
3 X - - - - - X X 
4 X - - - - - X X 
5 X - - - - - X X 
6 X - - - - - X X 
7 X - - - - - X X

BOUND1(6) bit
1<<6:   	64  1000000
  0 1 2 3 4 5 6 7
0 - - - - - - - Q  aBoard[0]=1
1 X Q - - - - X X  aBoard[1]=bit=1<<BOUND1(6)
2 X - - - - - X X 
3 X - - - - - X X 
4 X - - - - - X X 
5 X - - - - - X X 
6 X - - - - - X X 
7 X - - - - - X X
*/
		aBoard[1]=bit=1<<BOUND1;

		Backtrack1(2,(2|bit)<<1,1|bit,bit>>1,BOUND1);
		BOUND1++;
	}


	TOPBIT=1<<SIZEE;				//128,10000000
/**
TOPBITは、
SIDEMASK,LASTMASK,ENDBITを算出するときcheckメソッドで
２７０度回転でクイーンの位置が左端にあるかどうかチェック
するときに使用されている	

TOPBIT  SIZEE(7)  bit
	   		1<<7:  		128      10000000	
  0 1 2 3 4 5 6 7
0 Q - - - - - - - 		
1 - - - - - - - - 	  
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
*/

	SIDEMASK=LASTMASK=TOPBIT|1; //TOPBIT|1 129: 10000001
/**
										TOPBIT 		10000000
TOPBIT
  0 1 2 3 4 5 6 7		
0 Q - - - - - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

SIDEMASK=LASTMASK=TOPBIT|1  10000001 
  0 1 2 3 4 5 6 7		
0 Q - - - - - - Q  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
*/

	ENDBIT=TOPBIT>>1;           //TOPBIT>>1 64: 01000000
/**
										ENDBIT 	TOPBIT>>1 
														64				01000000
TOPBIT
  0 1 2 3 4 5 6 7		
0 Q - - - - - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

ENDBIT=TOPBIT>>1
  0 1 2 3 4 5 6 7		
0 - Q - - - - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
*/	

	BOUND1=1;
	int BOUND2=SIZE-2;;
	while(BOUND1<BOUND2){
		/*0行目:000001110(選択)*/
		aBoard[0]=bit=1<<BOUND1;
/**
aBoard[0]=bit=1<<BOUND1(1)
     BOUND1 bit
	1<<1        2   10
  0 1 2 3 4 5 6 7		
0 - - - - - - Q - 	
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

aBoard[0]=bit=1<<BOUND1(2)
     BOUND1 bit
	1<<2        4   100
  0 1 2 3 4 5 6 7		
0 - - - - - Q - - 	
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

aBoard[0]=bit=1<<BOUND1(3)
		 BOUND1   bit
	1<<3    		8 	1000
  0 1 2 3 4 5 6 7		
0 - - - - Q - - - 	
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

aBoard[0]=bit=1<<BOUND1(4)
		 BOUND1		bit
	1<<4   			16	10000
  0 1 2 3 4 5 6 7		
0 - - - Q - - - - 	
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

aBoard[0]=bit=1<<BOUND1(5)
		 BOUND1		bit
	1<<5  			32	100000
  0 1 2 3 4 5 6 7		
0 - - Q - - - - - 	
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
*/
		Backtrack2(1,bit<<1,bit,bit>>1,BOUND1,BOUND2);
		LASTMASK|=LASTMASK>>1|LASTMASK<<1;
/**
SIDEMASK=LASTMASK=TOPBIT|1  10000001 
  0 1 2 3 4 5 6 7		
0 Q - - - - - - Q  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

LASTMASK|=LASTMASK>>1|LASTMASK<<1
  0 1 2 3 4 5 6 7		
0 - Q - - - - Q -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
*/

		ENDBIT>>=1;
/**
TOPBIT
  0 1 2 3 4 5 6 7		
0 Q - - - - - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

ENDBIT=TOPBIT>>1
  0 1 2 3 4 5 6 7		
0 - Q - - - - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

ENDBIT>>=1 (１回目のループ）
  0 1 2 3 4 5 6 7		
0 - - Q - - - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

ENDBIT>>=1 (２回目のループ）
  0 1 2 3 4 5 6 7		
0 - - - Q - - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

ENDBIT>>=1 (３回目のループ）
  0 1 2 3 4 5 6 7		
0 - - - - Q - - -  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -

*/
		BOUND1++;
		BOUND2--;
	}
	UNIQUE=COUNT8+COUNT4+COUNT2;
	TOTAL=COUNT8*8+COUNT4*4+COUNT2*2;
}
int main(){
  SIZE=8;
	NQueens();
  printf("count:%d\n",count);
	printf("%2d:%16d%16d\n", SIZE, TOTAL, UNIQUE);
  return 0;
}
