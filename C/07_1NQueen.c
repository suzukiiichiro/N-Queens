#include <stdio.h>
#include <time.h>
#define MINSIZE 0
#define MAXSIZE 8
int c=1 ; //# c:count
int pos[MAXSIZE];
void NQueen1(int i,int s)
{
  int j ; //# s:size
  for(j=0;j<s;j++){
      pos[i]=j ;
      if ((i==s-1)){ 
        printf("%d",c++);
        for(int x=0;x<s;x++){
          printf("%d",pos[x]);
        }
        printf("\n");
      }else{
        NQueen1(i+1,s);
      }
  }  
}

int main(void)
{
NQueen1(MINSIZE,MAXSIZE);
}
