#include <stdio.h>

int TOTAL=0;
int UNIQUE=0;
int size=8;

void NQueenD(int size,int row){
  int board[size];
  int* tmpBoard;
  int mask=(1<<size)-1;
  int bitmap=mask;
  int bit;
  int down[size];
  int right[size];
  int left[size];
  for(int i=0;i<size;i++){
    board[i]=-1; 
  }
  tmpBoard=board+1;             
  
  while(1){
    if(bitmap){
      bit=(-bitmap&bitmap); 
      bitmap&=~bit;
      if(row==(size-1)){
        TOTAL++;
        tmpBoard--;
        bitmap=*tmpBoard;
        --row;
        continue;
      }else{
        int n=row++;
        left[row]=(left[n]|bit)<<1;
        down[row]=down[n]|bit;
        right[row]=(right[n]|bit)>>1;
        *tmpBoard=bitmap;
        tmpBoard++;
        bitmap=mask&~(left[row]|down[row]|right[row]);
        continue;
      }
    }else{ 
      tmpBoard--;
      bitmap=*tmpBoard;
      if(tmpBoard==board){ break ; }
      --row;
      continue;
    }
  }
}

int main(){
  NQueenD(size,0);
  printf("TOTAL: %d",TOTAL);
  return 0;
}
