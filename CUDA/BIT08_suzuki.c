// gcc BIT08.c && ./a.out ;
/**
 * クイーンを中央に配置する場合には、１行目を右半分だけ実行して実行結果
 * を２倍にする（BIT05）
 * 0,1行目でクイーンを配置する位置をBOUND1で制御 （BIT06）
 * 0,1行目でクイーンを配置する位置をBOUND2でも制御（BIT07）
 *  
BIT01.c 行も斜めも考慮せずに配置できる可能性を出力
   0 1 2 3 4
 0 Q - - - -
 1 Q - - - -
 2 Q - - - -
 3 Q - - - -
>4 Q - - - -
N=5 No.3125 Step.3905 backtrack(),+189,

BIT02.c 行に一つだけ配置できるように考慮したプログラム
   0 1 2 3 4
 0 Q - - - -
 1 - Q - - -
 2 - - Q - -
 3 - - - Q -
>4 - - - - Q
N=5 No.120 Step.325 backtrack(),+174,

BIT03.c 斜めを考慮して配置できる可能性を出力 一般的なバックトラック
   0 1 2 3 4
 0 Q - - - -
 1 - - Q - -
 2 - - - - Q
 3 - Q - - -
>4 - - - Q -
N=5 No.10 Step.53 backtrack(),+485,

BIT04.c 最上部右半分だけ実行 偶数の場合は実行結果を２倍。奇数と偶数を考慮
   0 1 2 3 4
 0 - - Q - -
 1 Q - - - -
 2 - - - Q -
 3 - Q - - -
>4 - - - - Q
N=5 No.10 Step.31 backtrack(),+489,

BIT05.c クイーンを中央に配置する場合、一行目を右半分だけ実行し、実行結果を２倍する。奇数と偶数を考慮
   0 1 2 3 4
 0 - - Q - -
 1 - - - - Q
 2 - Q - - -
 3 - - - Q -
>4 Q - - - -
N=5 No.5 Step.27 backtrack(),+489,

BIT06.c  0,1行目でクイーンを配置する位置をBOUND1で制御
   0 1 2 3 4
 0 - - Q - -
 1 - - - - Q
 2 - Q - - -
 3 - - - Q -
>4 Q - - - -
N=5 No.5 Step.27 backtrack(),+537,

BIT07.c 0,1行目でクイーンを配置する位置をBOUND2でも制御（BIT07）
   0 1 2 3 4
 0 - - Q - -
 1 - - - - Q
 2 - Q - - -
 3 - - - Q -
>4 Q - - - -
N=5 No.5 Step.27 backtrack(),+548,
*/

#include <stdio.h>
#include <string.h>


int size;       //ボードサイズ
int mask;       //連続する１ビットのシーケンス N=8: 11111111
int count=0;      //見つかった解
int aBoard[17];  //表示用配列
// 
//１６進数を２進数に変換
void con(int decimal){
  int _decimal=decimal;
  int binary=0;
  int base=1;
  while(decimal>0){
    binary=binary+(decimal%2)*base;
    decimal=decimal/2;
    base=base*10;
  }
  printf("16進数:\t%d\t2進数:\t%d\n",_decimal,binary);
}
//
//ボード表示用
int step=0;
char pause[32]; 
void Display(int y,int LINE,const char* FUNC,int left,int down,int right,int BOUND1,int BOUND2,int flg_2) {
  printf("\n");
  for (int row=0; row<size; row++) {
    if(row==0){ printf("   ");
      for(int col=0;col<size;col++){ printf("%d ",col); } 
      printf("\n");
    }
    if(row==y){ printf(">%d ",row); }
    else{ printf(" %d ",row); }
    int bitmap = aBoard[row];
    char s;
    int cnt=size-1;
    for (int bit=1<<(size-1); bit; bit>>=1){
      if(row>y){ s='-'; }
      else{ s=(bitmap & bit)? 'Q': '-'; }
      if(row==y+1){
        if((bit& left)){ s='x'; }
        if((bit&right)){ s='x'; }
        if((bit& down)){ s='x'; }
      }
      //backtrack1の時の枝狩り
      if((row<=BOUND1&&BOUND2==0)&&flg_2==1){
        if(cnt==1){
          s='2';
        }
      }
      cnt--;
      printf("%c ", s);
    }
    printf("\n");
  }
  step++;
  if(y==size-1){
    printf("N=%d No.%d Step.%d %s(),+%d,\n\n",size,count+1,step,FUNC,LINE);
  }
  if(strcmp(pause, ".") != 10){ fgets(pause,sizeof(pause),stdin); }
}
// y:これまでに配置できたクイーンの数
void backtrack(int y,int left,int down,int right,int BOUND1,int BOUND2){
  int bitmap=0;
  int bit=0;
  int flg_2;            //表示用
  //行・斜めを考慮して配置できる可能性を出力
  bitmap=mask&~(left|down|right); 
  if(y==size-1){
    if(bitmap){
      aBoard[y]=bitmap; //表示用
      count++;
    }
  }else{
    /**
     * 枝刈り（１）
     */
    if(BOUND1==1){       
      bitmap|=2;
      bitmap^=2;
      flg_2=1;          //表示用
      Display(y-1,__LINE__,__func__,
          (left|bit),(down|bit),(right|bit),
          BOUND1,BOUND2,
          flg_2         //flg_2
      );                //表示用
    }
    while(bitmap){
      bit=-bitmap&bitmap;
      // ここでは配置可能なパターンがひとつずつ生成される(bit) 
      bitmap^=bit;
      aBoard[y]=bit;    // 表示用
      Display(y,__LINE__,__func__,
          (left|bit)<<1,(down|bit),(right|bit)>>1,
          BOUND1,BOUND2,
          flg_2         //flg_2
      );                //表示用
      backtrack(y+1,(left|bit)<<1,(down|bit),(right|bit)>>1,BOUND1,BOUND2);
    }
  }
}
/**
  最上部の右半分だけ実行
  偶数の場合は実行結果を２倍にする。
  奇数の場合はクイーンを中央に配置する場合には、１行目を右半分だけ実行して実行結果を２倍にする
*/
void NQueen(void){
  int bitmap,bit,down,right,left;
  /*右半分限定0行目:000001111*/
  int BOUND1=0;
  int BOUND2=size-1;
  //ここではBOUND1は0行目にクイーンを置く場所として使用する
  //右端から左端へ向けてBOUND1を一つづず動かしていく
  //0行目右半分まで行ったら終了
  while(BOUND1<BOUND2){
    bit=1<<BOUND1;
    aBoard[0]=bit;      // 表示用
    Display(0,__LINE__,__func__,bit<<1,bit,bit>>1,
        BOUND1,BOUND2,
        0               //flg_2
    );                  //表示用
    backtrack(1,bit<<1,bit,bit>>1,BOUND1,BOUND2);
    BOUND1++;
    BOUND2--;
  }
  /*奇数の中央0行目:000010000*/
  //クイーンを中央に配置する場合は1行目の処理を右半分にしないと左右反転２パターンずつできる
  if(size&1){ //sizeが奇数だったら
    //奇数の場合はBOUND1がクイーンの位置が中央になっている
    bit=1<<BOUND1;
    down=bit;
    right=bit>>1;
    left=bit<<1;
    aBoard[0]=bit;      //表示用
    Display(0,__LINE__,__func__,bit<<1,bit,bit>>1,
        BOUND1,BOUND2,
        0               //flg_2
    );                  //表示用
    //1行目については右側半分だけ実行する
    //ここではBOUND1は1行目のクイーンの配置のために使用する
    BOUND1=0;
    //1行目は右半分までしか置けない
    BOUND2=size-1;
    while(BOUND1<BOUND2){
      bit=1<<BOUND1;
      aBoard[1]=bit;    //表示用
      Display(1,__LINE__,__func__,
          (left|bit)<<1,(down|bit),(right|bit)>>1,
          BOUND1,BOUND2,
          0             //flg_2
      );                //表示用
      backtrack(2,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,BOUND2);
      BOUND1++;
      BOUND2--;
    }
  }
  count*=2;/*左右反転パターンを考慮*/
}
int main(){
  size=10;
  mask=(1<<size)-1;
  NQueen();
  printf("COUNT:%d\n",count);
  return 0;
}
