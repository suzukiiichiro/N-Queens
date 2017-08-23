/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 <>30. デバッグトレース


*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include "unistd.h"


void create(int y,int si,int d,int a[]){
 if(y==d){
      for(int j=0;j<si;j++){//backtrack2の集計
        printf("create:%d:%d  %d,%d\n",y,j,a[1],a[2]);
        switch(y-1){
          case 1: ; break;
          case 2: ; break;
          case 3: ; break;
          case 4: ; break;
          case 5: ; break;
          case 6: ; break;
          case 7: ; break;
          case 8: ; break;
          case 9: ; break;
          case 10: ; break;
        }
      }
      for(int j=0;j<si;j++){//backtrack2の集計
        printf("join:%d:%d\n",y,j);
      }
 }else{
    for(int j=0;j<si;j++){//backtrack2の集計
      a[y]=j;
      create(y+1,si,d,a); 
    }
 } 

}
int main(void){
  int a[8];
  create(1,8,3,a);
  return 0;
}
