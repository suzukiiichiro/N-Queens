#include <stdio.h>
#include <time.h>
#define MINSIZE 0
#define MAXSIZE 8
int c=1 ; //# c:count
int pos[MAXSIZE];
int flag_a[MAXSIZE];//C言語のboolean の使い方不明
void NQueen2(int i,int s)
{
  int j ; //# s:size
  for(j=0;j<s;j++){
    if(! flag_a[j]){
      pos[i]=j ;
      if(i==s-1){
        printf("%d",c++);
        for(i=0;i<s;i++){
          printf("%d",pos[i]);
        }
        printf("\n");
      }else{
        flag_a[j]=1;         
        NQueen2(i+1,s);
        flag_a[j]=0; 
      }
    }
  }
}
int main(void)
{
NQueen2(MINSIZE,MAXSIZE);
}
