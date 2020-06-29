// gcc BIT03.c && ./a.out ;

#include <stdio.h>

/**
 * 行・斜めを考慮して配置できる可能性を出力 
 */

/**
    N=5 no.1
    - - - - Q 
    - - Q - - 
    Q - - - - 
    - - - Q - 
    - Q - - - 

    N=5 no.2
    - - - - Q 
    - Q - - - 
    - - - Q - 
    Q - - - - 
    - - Q - - 

    N=5 no.3
    - - - Q - 
    - Q - - - 
    - - - - Q 
    - - Q - - 
    Q - - - - 

    N=5 no.4
    - - - Q - 
    Q - - - - 
    - - Q - - 
    - - - - Q 
    - Q - - - 

    N=5 no.5
    - - Q - - 
    - - - - Q 
    - Q - - - 
    - - - Q - 
    Q - - - - 

    N=5 no.6
    - - Q - - 
    Q - - - - 
    - - - Q - 
    - Q - - - 
    - - - - Q 

    N=5 no.7
    - Q - - - 
    - - - - Q 
    - - Q - - 
    Q - - - - 
    - - - Q - 

    N=5 no.8
    - Q - - - 
    - - - Q - 
    Q - - - - 
    - - Q - - 
    - - - - Q 

    N=5 no.9
    Q - - - - 
    - - - Q - 
    - Q - - - 
    - - - - Q 
    - - Q - - 

    N=5 no.10
    Q - - - - 
    - - Q - - 
    - - - - Q 
    - Q - - - 
    - - - Q - 
    count:10
*/
int size;       //ボードサイズ
int mask;       //連続する１ビットのシーケンス N=8: 11111111
int count;      //見つかった解
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
void Display(void) {
    int  y, bitmap, bit;
    printf("\nN=%d no.%d\n", size, count);
    for (y=0; y<size; y++) {
        bitmap = aBoard[y];
        for (bit=1<<(size-1); bit; bit>>=1)
            printf("%s ", (bitmap & bit)? "Q": "-");
        printf("\n");
    }
}
// y:これまでに配置できたクイーンの数
void backtrack(int y,int left,int down,int right){
  int bitmap=0;
  int bit=0;
  if(y==size){
    count++;
    Display(); //表示
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
      backtrack(y+1,(left|bit)<<1,(down|bit),(right|bit)>>1);
    }
  }
}
int main(){
  size=5;
  mask=(1<<size)-1;
  backtrack(0,0,0,0);
  printf("count:%d\n",count);
  return 0;
}
