// gcc BIT05.c && ./a.out ;

#include <stdio.h>
#include <string.h>

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
int step;
char pause[32]; 
int fault=0;
int flg_2=0;
int flg_s=0;
int flg_sg=0;
int flg_m=0;
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
  printf("%s 10進数:\t%d\t2進数:\t%d\n",c,_decimal,binary);
}
//
//ボード表示用
void Display(int y,int BOUND1,int BOUND2,int MODE,int LINE,const char* FUNC,int C,int bm,int left,int down,int right){
  //MODE=-2 TOPBIT,ENDBITを優先する
  //MODE=0 Qを優先する
  //MODE=-1 最下段枝狩りで枝狩りされる場合
  //MODE=1 Backtrack1の枝狩りフラグをオンにする
  //MODE=2 Backtrack2上部枝狩りのフラグをオンにする
  //MODE=3 Backtrack2 下部枝狩りのフラグをオンにする
  //MODE=4 Backtrack2 最下段枝狩りのフラグをオンにする
    int  row, bitmap, bit;
    char* s;
    switch(MODE){
    case 1:
    //MODE=1 Backtrack1の枝狩りフラグをオンにする
      flg_2=1;
    case 2:
    //MODE=2 Backtrack2上部枝狩りのフラグをオンにする
      flg_s=1;
    case 3:
    //MODE=3 Backtrack2 下部枝狩りのフラグをオンにする
      flg_sg=1;
    case 4:
    //MODE=4 Backtrack2 最下段枝狩りのフラグをオンにする
      flg_m=1;
    default:
      printf("");
    }
    if(fault>y){
      printf("####枝狩り####\n\n");
    }
    printf("Line:%d,Func:%s,Count:%d:Step.%d\n",LINE,FUNC,C,++step);
    if(BOUND2 !=0){
      con("SIDEMASK",SIDEMASK);
      con("LASTMASK",LASTMASK);
      con("TOPBIT",TOPBIT);
      con("ENDBIT",ENDBIT);
    }
    printf("\nN=%d no.%d BOUND1:%d:BOUND2:%d:y:%d\n", SIZE, ++count,BOUND1,BOUND2,y);
    int row_cnt=0;
    for (row=0; row<SIZE; row++) {
      if(row==0){
        printf("   ");
        for(int col=0;col<SIZE;col++){
          printf("%d ",col);
        }
        printf("\n");
      }
      if(row==y){
        printf(">%d ",row);
      }else{
        printf(" %d ",row);
      }
        bitmap = aBoard[row];
        int cnt=SIZEE;
        for (bit=1<<(SIZEE); bit; bit>>=1){
            int mb=1<<cnt;
            if(row_cnt>y){
              s="-";
            }else{
              s=(bitmap & bit)? "Q": "-";
            }
            //MASKの処理
            if(row_cnt==y+1){
              if(!(bit&bm)){
               s="x";
              }
              if(MODE !=-2){ 
                if(aBoard[(row-1)] & bit){
                  s="D";
                }
                if(((aBoard[(row-1)] <<1)& bit)){
                  s="L";
                }
                if(((aBoard[(row-1)] >>1)& bit)){
                  s="R";
                }
              }
            
            }
            //backtrack1の時の枝狩り
            if((row<=BOUND1&&BOUND2==0)&&flg_2==1){
              if(cnt==1){
               s="2";
              }
            }
            //SIDEMASKの処理
            if(((row<BOUND1)&&BOUND2!=0)&&flg_s==1){
             if(cnt==0||cnt==SIZEE){
              s="S";
             } 
            }
            if(((row>BOUND2)&&BOUND2!=0)&&flg_sg==1){
             if(cnt==0||cnt==SIZEE){
              s="S";
             } 
            }
            //最終行の処理
            if(row==SIZEE&&BOUND2!=0){
            //LASTMASKの処理
              if((LASTMASK&bit)&&flg_m==1){
               s="M"; 
              }
            //ENDBITの処理
              if(ENDBIT&bit&&MODE==5){
               s="E";
              }

            }
            //TOPBITの処理
            if(row==BOUND1&&cnt==SIZEE&&BOUND2!=0&&MODE==5){
              s="T";
            }
            cnt--;
            printf("%s ", s);
        }
        printf("\n");
        row_cnt++;
    }
       if(C>0){
         flg_2=0;
         flg_s=0;
         flg_m=0;
         fault=0;
         printf("####処理完了####\n");
       }else{ 
        if(fault<y){
          printf("\n");
        }
       }
       if(strcmp(pause, ".") != 10){
         fgets(pause,sizeof(pause),stdin);
       }
       fault=y;
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
			Display(SIZEE,BOUND1,BOUND2,0,__LINE__,__func__,2,0,0,0,0); //表示用
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
			Display(SIZEE,BOUND1,BOUND2,0,__LINE__,__func__,4,0,0,0,0); //表示用
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
	Display(SIZEE,BOUND1,BOUND2,0,__LINE__,__func__,8,0,0,0,0); //表示用
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
			        Display(y-1,BOUND1,BOUND2,4,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit)); //表示用
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
			Display(y-1,BOUND1,BOUND2,2,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit)); //表示用
/**
兄　ここの枝刈りの遷移をおねがい
*/
		}else if(y==BOUND2){/*下部サイド枝刈り*/
			if(!(down&SIDEMASK)){
			  Display(y,BOUND1,BOUND2,-1,__LINE__,__func__,0,MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1),(left|bit)<<1,(down|bit),(right|bit)>>1); //表示用
        return;
      }
			if((down&SIDEMASK)!=SIDEMASK)bitmap&=SIDEMASK;
			Display(y-1,BOUND1,BOUND2,3,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit)); //表示用
/**
兄　ここの枝刈りの遷移をおねがい
*/
		}
		while(bitmap){
			bitmap^=aBoard[y]=bit=-bitmap&bitmap;
			Display(y,BOUND1,BOUND2,0,__LINE__,__func__,0,MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1),(left|bit)<<1,(down|bit),(right|bit)>>1); //表示用
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
  最終行にクイーンを配置する
*/
			COUNT8++;
			Display(SIZEE,BOUND1,0,1,__LINE__,__func__,8,0,0,0,0);  //表示用
		}
	}else{
    //y=2の時はこの枝狩りは不要。最適化できないか検討する
		if(y<BOUND1){/*枝刈り : 斜軸反転解の排除*/
			bitmap|=2;
			bitmap^=2;
		        Display(y-1,BOUND1,0,1,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit)); //表示用
/**
  右から２列目を枝狩りする
  ex 10001010->10001000
  一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つの角にクイーン
  を置くことはできないので、ユニーク解であるかどうかを判定するには、右上角から
  左下角を通る斜軸で反転させたパターンとの比較だけになります。突き詰めれば、上
  から２行目のクイーンの位置が右から何番目にあるかと、右から２列目のクイーンの
  位置が上から何番目にあるかを比較するだけで判定することができます。この２つの
  値が同じになることはないからです。
  結局、再帰探索中において y<BOUND1の時に右から２列目に枝狩りを入れておけばユニ
  ーク解になる。  
  右上角から左下角の斜軸で反転から考慮するとy=<BOUND1 になるが
  y=BOUND1の時は利き筋に当たるので枝狩りには入れない

	  0 1 2 3 4 5 6 7 BOUND1(2)
	0 - - - - - - - Q  
	1 - - - - - Q - -  
	2 - - - - - - X - 
	3 - - - - - - - - 
	4 - - - - - - - - 
	5 - - - - - - - - 
	6 - - - - - - - - 
	7 - - - - - - - -

	  0 1 2 3 4 5 6 7 BOUND1(3)
	0 - - - - - - - Q  
	1 - - - - Q - - -  
	2 - - - - - - X - 
	3 - - - - - - X - 
	4 - - - - - - - - 
	5 - - - - - - - - 
	6 - - - - - - - - 
	7 - - - - - - - -

	  0 1 2 3 4 5 6 7 BOUND1(4)
	0 - - - - - - - Q  
	1 - - - Q - - - -  
	2 - - - - - - X - 
	3 - - - - - - X - 
	4 - - - - - - X - 
	5 - - - - - - - - 
	6 - - - - - - - - 
	7 - - - - - - - -

	  0 1 2 3 4 5 6 7 BOUND1(5)
	0 - - - - - - - Q  
	1 - - Q - - - - -  
	2 - - - - - - X - 
	3 - - - - - - X - 
	4 - - - - - - X - 
	5 - - - - - - X - 
	6 - - - - - - - - 
	7 - - - - - - - -

	  0 1 2 3 4 5 6 7 BOUND1(6)
	0 - - - - - - - Q  
	1 - Q - - - - - -  
	2 - - - - - - X - 
	3 - - - - - - X - 
	4 - - - - - - X - 
	5 - - - - - - X - 
	6 - - - - - - X - 
	7 - - - - - - - -

*/
		}
		while(bitmap){
			bitmap^=aBoard[y]=bit=-bitmap&bitmap;
/**
	bitmap^=aBoard[y]=bit=-bitmap&bitmap;
  クイーンの配置　-bitmap&bitmap で一番右端にクイーンを置く
  bitmap=11110000 -> bitmap=11100000 
                     aBoard[y]=00010000

*/
		  Display(y,BOUND1,0,0,__LINE__,__func__,0,MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1),(left|bit)<<1,(down|bit),(right|bit)>>1); //表示用
			Backtrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1);
		}
	}
}
//
void NQueens(void) {
	int  bit;
	COUNT8=COUNT4=COUNT2=0;
	SIZEE=SIZE-1;

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
	MASK=(1<<SIZE)-1;       //255,11111111


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
        Display(0,2,0,-2,__LINE__,__func__,0,MASK,0,0,0); //表示用
	aBoard[0]=1;						//1,  00000001
        Display(0,2,0,-1,__LINE__,__func__,0,MASK&~((1)<<1|(1)|(1)>>1),1<<1,(1),1>>1); //表示用


	/*0行目:000000001(固定)*/
	/*1行目:011111100(選択)*/
	int BOUND1=2;
	//for(BOUND1=2;BOUND1<SIZEE;BOUND1++){
	while(BOUND1<SIZEE){

/**
	aBoard[1]=bit=1<<BOUND1; についての遷移

	BOUND1(2) 	bit
	1<<2:  			4      100	
		0 1 2 3 4 5 6 7
	0 - - - - - - - Q  aBoard[0]=1
	1 X - - - - Q - X  aBoard[1]=bit=1<<BOUND1(2)
	2 - - - - - - X X 
	3 - - - - - - - X 
	4 - - - - - - - X 
	5 - - - - - - - X 
	6 - - - - - - - X 
	7 - - - - - - - X

	BOUND1(3)  	bit
	1<<3:				8     1000
		0 1 2 3 4 5 6 7
	0 - - - - - - - Q  aBoard[0]=1
	1 X - - - Q - - X  aBoard[1]=bit=1<<BOUND1(3)
	2 - - - - - - X X 
	3 - - - - - - X X 
	4 - - - - - - - X 
	5 - - - - - - - X 
	6 - - - - - - - X 
	7 - - - - - - - X

	BOUND1(4) bit
	1<<4:    	16    10000
		0 1 2 3 4 5 6 7
	0 - - - - - - - Q  aBoard[0]=1
	1 X - - Q - - - X  aBoard[1]=bit=1<<BOUND1(4)
	2 - - - - - - X X 
	3 - - - - - - X X 
	4 - - - - - - X X 
	5 - - - - - - - X 
	6 - - - - - - - X 
	7 - - - - - - - X

	BOUND1(5) bit
	1<<5:    	32   100000
		0 1 2 3 4 5 6 7
	0 - - - - - - - Q  aBoard[0]=1
	1 X - Q - - - - X  aBoard[1]=bit=1<<BOUND1(5)
	2 - - - - - - X X 
	3 - - - - - - X X 
	4 - - - - - - X X 
	5 - - - - - - X X 
	6 - - - - - - - X 
	7 - - - - - - - X

	BOUND1(6) bit
	1<<6:   	64  1000000

		0 1 2 3 4 5 6 7
	0 - - - - - - - Q  aBoard[0]=1
	1 X Q - - - - - X  aBoard[1]=bit=1<<BOUND1(6)
	2 - - - - - - X X 
	3 - - - - - - X X 
	4 - - - - - - X X 
	5 - - - - - - X X 
	6 - - - - - - X X 
	7 - - - - - - - X
*/
		aBoard[1]=bit=1<<BOUND1;
		Display(1,BOUND1,0,0,__LINE__,__func__,0,MASK&~((bit|1)<<1|(bit|1)|(bit|1)>>1),(bit)<<1,(bit),(bit)>>1); //表示用
    //printf("BOUND1:%d\n",BOUND1);
    //con("aBoard[1]",aBoard[1]);

/**
		Backtrack1の挙動については Backtrack1()を参照
*/
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
                //盤面をクリアにする
		/*0行目:000001110(選択)*/
		aBoard[0]=bit=1<<BOUND1;
		Display(0,BOUND1,BOUND2,0,__LINE__,__func__,0,MASK&~(bit<<1|bit|bit>>1),bit<<1,bit,bit>>1); //表示用
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

*/
/**
		Backtrack2の挙動については Backtrack2()を参照
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
BOUND1->BOUND2切り替わり時(１回目のループ）
LASTMASK|=LASTMASK>>1|LASTMASK<<1 111000011  
11000011が正しい
  0 1 2 3 4 5 6 7		
0 Q Q - - - - Q Q  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
BOUND2->BOUND3切り替わり時(２回目のループ）
LASTMASK|=LASTMASK>>1|LASTMASK<<1 1111100111  
11100111が正しい
  0 1 2 3 4 5 6 7		
0 Q Q Q - - Q Q Q  
1 - - - - - - - - 
2 - - - - - - - - 
3 - - - - - - - - 
4 - - - - - - - - 
5 - - - - - - - - 
6 - - - - - - - - 
7 - - - - - - - -
BOUND3->BOUND4切り替わり時(３回目のループ）
LASTMASK|=LASTMASK>>1|LASTMASK<<1 11111111111  
11111111が正しい
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
  SIZE=6;
	NQueens();
  printf("count:%d\n",count);
	printf("%2d:%16d%16d\n", SIZE, TOTAL, UNIQUE);
  return 0;
}
