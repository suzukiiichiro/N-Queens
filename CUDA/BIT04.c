// gcc BIT04.c && ./a.out ;
/**
最上部の右半分だけ実行。偶数の場合は実行結果を２倍にする。
奇数の場合（クイーンを中央に配置する場合）は２倍にしない
*/

#include <stdio.h>
#include <string.h>

/**
       0 1 2 3 4 
    >0 - - - - Q 
     1 - - - x x 
     2 - - - - - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - - Q 
    >1 - - Q - - 
     2 - x x x x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - - Q 
     1 - - Q - - 
    >2 Q - - - - 
     3 x x x - x 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - - Q 
     1 - - Q - - 
     2 Q - - - - 
    >3 - - - Q - 
     4 x - x x x 

       0 1 2 3 4 
     0 - - - - Q 
     1 - - Q - - 
     2 Q - - - - 
     3 - - - Q - 
    >4 - Q - - - 
    N=5 No.1 Step.5 backtrack(),+484,


       0 1 2 3 4 
     0 - - - - Q 
    >1 - Q - - - 
     2 x x x - x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - - Q 
     1 - Q - - - 
    >2 - - - Q - 
     3 - x x x x 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - - Q 
     1 - Q - - - 
     2 - - - Q - 
    >3 Q - - - - 
     4 x x - x x 

       0 1 2 3 4 
     0 - - - - Q 
     1 - Q - - - 
     2 - - - Q - 
     3 Q - - - - 
    >4 - - Q - - 
    N=5 No.2 Step.9 backtrack(),+484,


       0 1 2 3 4 
     0 - - - - Q 
    >1 Q - - - - 
     2 x x x - x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - - Q 
     1 Q - - - - 
    >2 - - - Q - 
     3 x x x x x 
     4 - - - - - 

       0 1 2 3 4 
    >0 - - - Q - 
     1 - - x x x 
     2 - - - - - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - Q - 
    >1 - Q - - - 
     2 x x x x - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - Q - 
     1 - Q - - - 
    >2 - - - - Q 
     3 x x - x x 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - Q - 
     1 - Q - - - 
     2 - - - - Q 
    >3 - - Q - - 
     4 - x x x x 

       0 1 2 3 4 
     0 - - - Q - 
     1 - Q - - - 
     2 - - - - Q 
     3 - - Q - - 
    >4 Q - - - - 
    N=5 No.3 Step.16 backtrack(),+484,


       0 1 2 3 4 
     0 - - - Q - 
    >1 Q - - - - 
     2 x x - x - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - Q - 
     1 Q - - - - 
    >2 - - - - Q 
     3 x - x x x 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - Q - 
     1 Q - - - - 
     2 - - - - Q 
    >3 - Q - - - 
     4 x x x x x 

       0 1 2 3 4 
     0 - - - Q - 
     1 Q - - - - 
    >2 - - Q - - 
     3 x x x x - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - - Q - 
     1 Q - - - - 
     2 - - Q - - 
    >3 - - - - Q 
     4 x - x x x 

       0 1 2 3 4 
     0 - - - Q - 
     1 Q - - - - 
     2 - - Q - - 
     3 - - - - Q 
    >4 - Q - - - 
    N=5 No.4 Step.22 backtrack(),+484,


       0 1 2 3 4 
    >0 - - Q - - 
     1 - x x x - 
     2 - - - - - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - Q - - 
    >1 - - - - Q 
     2 x - x x x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - Q - - 
     1 - - - - Q 
    >2 - Q - - - 
     3 x x x - x 
     4 - - - - - 

       0 1 2 3 4 
     0 - - Q - - 
     1 - - - - Q 
     2 - Q - - - 
    >3 - - - Q - 
     4 - x x x x 

       0 1 2 3 4 
     0 - - Q - - 
     1 - - - - Q 
     2 - Q - - - 
     3 - - - Q - 
    >4 Q - - - - 
    N=5 No.5 Step.27 backtrack(),+484,


       0 1 2 3 4 
     0 - - Q - - 
    >1 Q - - - - 
     2 x x x - x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - - Q - - 
     1 Q - - - - 
    >2 - - - Q - 
     3 x - x x x 
     4 - - - - - 

       0 1 2 3 4 
     0 - - Q - - 
     1 Q - - - - 
     2 - - - Q - 
    >3 - Q - - - 
     4 x x x x - 

       0 1 2 3 4 
     0 - - Q - - 
     1 Q - - - - 
     2 - - - Q - 
     3 - Q - - - 
    >4 - - - - Q 
    N=5 No.6 Step.31 backtrack(),+484,


       0 1 2 3 4 
    >0 - Q - - - 
     1 x x x - - 
     2 - - - - - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - Q - - - 
    >1 - - - - Q 
     2 - x - x x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - Q - - - 
     1 - - - - Q 
    >2 - - Q - - 
     3 - x x x x 
     4 - - - - - 

       0 1 2 3 4 
     0 - Q - - - 
     1 - - - - Q 
     2 - - Q - - 
    >3 Q - - - - 
     4 x x x - x 

       0 1 2 3 4 
     0 - Q - - - 
     1 - - - - Q 
     2 - - Q - - 
     3 Q - - - - 
    >4 - - - Q - 
    N=5 No.7 Step.36 backtrack(),+484,


       0 1 2 3 4 
     0 - Q - - - 
     1 - - - - Q 
    >2 Q - - - - 
     3 x x x - x 
     4 - - - - - 

       0 1 2 3 4 
     0 - Q - - - 
     1 - - - - Q 
     2 Q - - - - 
    >3 - - - Q - 
     4 x x x x x 

       0 1 2 3 4 
     0 - Q - - - 
    >1 - - - Q - 
     2 - x x x x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 - Q - - - 
     1 - - - Q - 
    >2 Q - - - - 
     3 x x - x x 
     4 - - - - - 

       0 1 2 3 4 
     0 - Q - - - 
     1 - - - Q - 
     2 Q - - - - 
    >3 - - Q - - 
     4 x x x x - 

       0 1 2 3 4 
     0 - Q - - - 
     1 - - - Q - 
     2 Q - - - - 
     3 - - Q - - 
    >4 - - - - Q 
    N=5 No.8 Step.42 backtrack(),+484,


       0 1 2 3 4 
    >0 Q - - - - 
     1 x x - - - 
     2 - - - - - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 Q - - - - 
    >1 - - - - Q 
     2 x - x x x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 Q - - - - 
     1 - - - - Q 
    >2 - Q - - - 
     3 x x x x x 
     4 - - - - - 

       0 1 2 3 4 
     0 Q - - - - 
    >1 - - - Q - 
     2 x - x x x 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 Q - - - - 
     1 - - - Q - 
    >2 - Q - - - 
     3 x x x x - 
     4 - - - - - 

       0 1 2 3 4 
     0 Q - - - - 
     1 - - - Q - 
     2 - Q - - - 
    >3 - - - - Q 
     4 x x - x x 

       0 1 2 3 4 
     0 Q - - - - 
     1 - - - Q - 
     2 - Q - - - 
     3 - - - - Q 
    >4 - - Q - - 
    N=5 No.9 Step.49 backtrack(),+484,


       0 1 2 3 4 
     0 Q - - - - 
    >1 - - Q - - 
     2 x x x x - 
     3 - - - - - 
     4 - - - - - 

       0 1 2 3 4 
     0 Q - - - - 
     1 - - Q - - 
    >2 - - - - Q 
     3 x - x x x 
     4 - - - - - 

       0 1 2 3 4 
     0 Q - - - - 
     1 - - Q - - 
     2 - - - - Q 
    >3 - Q - - - 
     4 x x x - x 

       0 1 2 3 4 
     0 Q - - - - 
     1 - - Q - - 
     2 - - - - Q 
     3 - Q - - - 
    >4 - - - Q - 
    N=5 No.10 Step.53 backtrack(),+484,


*/
int size;       //ボードサイズ
int mask;       //連続する１ビットのシーケンス N=8: 11111111
int count=0;      //見つかった解
int aBoard[8];  //表示用配列
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
void Display(int y,int LINE,const char* FUNC,int left,int down,int right) {
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
    for (int bit=1<<(size-1); bit; bit>>=1){
      if(row>y){ s='-'; }
      else{ s=(bitmap & bit)? 'Q': '-'; }
      if(row==y+1){
        if((bit& left)){ s='x'; }
        if((bit&right)){ s='x'; }
        if((bit& down)){ s='x'; }
      }
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
void backtrack(int y,int left,int down,int right){
  int bitmap=0;
  int bit=0;
  if(y==size){
    count++;
  }else{
    //OR 結果をビット反転
    // mask:11111111 255
    // left 0: down 1: right 0
    // 11111110 254

    /**
     * 斜めを考慮せず、行にクイーンを一つだけ配置できるように考慮
     */
    //bitmap=mask&~(left|down|right);
    //bitmap=mask&~(0); //行も列も斜めも考慮せず配置できる可能性を出力
    //bitmap=mask&~(down); //行だけを考慮して配置できる可能性を出力
    bitmap=mask&~(left|down|right); //行・斜めを考慮して配置できる可能性を出力

    while(bitmap){
      // nを２進法で表したときの一番下位のONビットがひとつだけ抽出される結果が
      // 得られるのです。極めて簡単な演算によって1ビット抽出を実現させているこ
      // とが重要です。
      //     00010110   22
      // AND 11101010  -22
      // ------------------
      //     00000010
      bit=-bitmap&bitmap;
      // ここでは配置可能なパターンがひとつずつ生成される(bit) 
      bitmap^=bit;
      aBoard[y]=bit;  // 表示用
      Display(y,__LINE__,__func__,(left|bit)<<1,(down|bit),(right|bit)>>1); //表示
      backtrack(y+1,(left|bit)<<1,(down|bit),(right|bit)>>1);
    }
  }
}
/**
  最上部の右半分だけ実行
  偶数の場合は実行結果を２倍にする。
  奇数の場合（クイーンを中央に配置する場合）は２倍にしない
*/
void NQueen(void){
  int bitmap,bit,down,right,left;
  /*右半分限定0行目:000001111*/
  bitmap=(1<<(size/2))-1;/*0行目の配置可能ビット*/
  while(bitmap){
    bit=-bitmap&bitmap;
    bitmap^=bit;
    aBoard[0]=bit;
    Display(0,__LINE__,__func__,bit<<1,bit,bit>>1); //表示
    backtrack(1,bit<<1,bit,bit>>1);
  }
  count*=2;/*左右反転パターンを考慮*/
  /*奇数の中央0行目:000010000*/
  //クイーンを中央に配置する場合は1行目の処理を右半分にしないと左右反転２パターンずつできる
  if(size&1){ //sizeが奇数だったら
    bitmap=(1<<(size/2));/*0行目の配置可能ビット*/
    while(bitmap){
      bit=-bitmap&bitmap;
      bitmap^=bit;
      aBoard[0]=bit;
      Display(0,__LINE__,__func__,bit<<1,bit,bit>>1); //表示
      backtrack(1,bit<<1,bit,bit>>1);
    }
  }
}
int main(){
  size=5;
  mask=(1<<size)-1;
	NQueen();
  printf("COUNT:%d\n",count);
  return 0;
}
