// gcc BIT01.c && ./a.out ;

#include <stdio.h>
#include <string.h>

/**
 * 行も斜めも考慮せず配置できる可能性を出力
 */

/**
    N=5 no.1
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - - Q 

    N=5 no.2
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - Q - 

    N=5 no.3
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - Q - - 

    N=5 no.4
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - Q - - - 

    N=5 no.5
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - - Q 
    Q - - - - 

    N=5 no.6
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - Q - 
    - - - - Q 

    N=5 no.7
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - Q - 
    - - - Q - 

    N=5 no.8
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - Q - 
    - - Q - - 

    N=5 no.9
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - Q - 
    - Q - - - 

    N=5 no.10
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - - Q - 
    Q - - - - 

    N=5 no.11
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - Q - - 
    - - - - Q 

    N=5 no.12
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - Q - - 
    - - - Q - 

    N=5 no.13
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - Q - - 
    - - Q - - 

    N=5 no.14
    - - - - Q 
    - - - - Q 
    - - - - Q 
    - - Q - - 
    - Q - - - 

＜＜省略＞＞

    N=5 no.3108
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - - Q - 
    - - Q - - 

    N=5 no.3109
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - - Q - 
    - Q - - - 

    N=5 no.3110
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - - Q - 
    Q - - - - 

    N=5 no.3111
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - Q - - 
    - - - - Q 

    N=5 no.3112
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - Q - - 
    - - - Q - 

    N=5 no.3113
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - Q - - 
    - - Q - - 

    N=5 no.3114
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - Q - - 
    - Q - - - 

    N=5 no.3115
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - Q - - 
    Q - - - - 

    N=5 no.3116
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - Q - - - 
    - - - - Q 

    N=5 no.3117
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - Q - - - 
    - - - Q - 

    N=5 no.3118
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - Q - - - 
    - - Q - - 

    N=5 no.3119
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - Q - - - 
    - Q - - - 

    N=5 no.3120
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - Q - - - 
    Q - - - - 

    N=5 no.3121
    Q - - - - 
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - - - Q 

    N=5 no.3122
    Q - - - - 
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - - Q - 

    N=5 no.3123
    Q - - - - 
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - - Q - - 

    N=5 no.3124
    Q - - - - 
    Q - - - - 
    Q - - - - 
    Q - - - - 
    - Q - - - 

    N=5 no.3125
    Q - - - - 
    Q - - - - 
    Q - - - - 
    Q - - - - 
    Q - - - - 
    count:3125

*/
int size;       //ボードサイズ
int mask;       //連続する１ビットのシーケンス N=8: 11111111
int count=1;      //見つかった解
int step=1;
char pause[32]; 
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
void Display(int y,int LINE,const char* FUNC) {
    printf("\nLine:%d,Func:%s,N=%d no.%d step.%d y:%d\n",LINE,FUNC, size, count,step,y);
    int row_cnt=0;
    for (int row=0; row<size; row++) {
        if(row==y){
          printf(">%d ",row);
        }else{
          printf(" %d ",row);
        }
        int bitmap = aBoard[row];
        int cnt=size-1;
        char* s;
        for (int bit=1<<(size-1); bit; bit>>=1){
         if(row_cnt>y){
           s="-";
         }else{
           s=(bitmap & bit)? "Q": "-";
         }
         cnt--;
         printf("%s ", s);
        }
        printf("\n");
        row_cnt++;
    }
    if(y==size-1){
      printf("####処理完了####\n");
    }
    if(strcmp(pause, ".") != 10){
         fgets(pause,sizeof(pause),stdin);
    }
}
// y:これまでに配置できたクイーンの数
void backtrack(int y,int left,int down,int right){
  int bitmap,bit;
  //bitmap=mask&~(left|down|right);
    bitmap=mask&~(0); //行も斜めも考慮せず配置できる可能性を出力
  if(y==size){
    count++;
  }else{
    //OR 結果をビット反転
    // mask:11111111 255
    // left 0: down 1: right 0
    // 11111110 254

    /**
     * 行も斜めも考慮せず配置できる可能性を出力
     */

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
      Display(y,__LINE__,__func__);
      step++;
      backtrack(y+1,(left|bit)<<1,(down|bit),(right|bit)>>1);
    }
  }
}
int main(){
  size=5; //サイズは５で
  mask=(1<<size)-1;
  backtrack(0,0,0,0);
  printf("count:%d\n",count);
  return 0;
}
