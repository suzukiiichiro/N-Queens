#include <stdio.h>
#include <time.h>

#define MAXSIZE 12

// 日本語

int iTotal=1 ;
int iUnique=0;
int iSize;
int colChk [2*MAXSIZE-1];
int diagChk[2*MAXSIZE-1];
int antiChk[2*MAXSIZE-1];
int aBoard[MAXSIZE];

void TimeFormat(clock_t utime, char *form) {
    int dd,hh,mm;
    float ftime,ss;
    ftime=(float)utime/CLOCKS_PER_SEC;
    mm=(int)ftime/60;
    ss=ftime-(float)(mm * 60);
    dd=mm/(24*60);
    mm=mm%(24*60);
    hh=mm/60;
    mm=mm%60;
    if (dd) sprintf(form,"%4d %02d:%02d:%05.2f",dd,hh,mm,ss);
    else if (hh) sprintf(form, "     %2d:%02d:%05.2f",hh,mm,ss);
    else if (mm) sprintf(form, "        %2d:%05.2f",mm,ss);
    else sprintf(form, "           %5.2f",ss);
}
void NQueen3(int row){
  if(row==iSize){
    iTotal++;
  }else{
    for(int col=0;col<iSize;col++){
      aBoard[row]=col ;
      if(colChk[col]==0 && diagChk[row-col+(iSize-1)]==0 && antiChk[row+col]==0){
        colChk[col]=diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=1; 
        NQueen3(row+1); 
        colChk[col]=diagChk[row-aBoard[row]+iSize-1]=antiChk[row+aBoard[row]]=0; 
      }
    }  
  }
}
int main(void) {
  clock_t st; 
  char t[20];
  printf("%s\n"," N:        Total       Unique        hh:mm:ss.ms");
  for(int i=2;i<=MAXSIZE;i++){
    iSize=i;
    iTotal=0; 
    iUnique=0; 
    for(int j=0;j<iSize;j++){
      aBoard[j]=j;
    }
    st=clock();
    NQueen3(0);
    TimeFormat(clock()-st,t);
    printf("%2d:%13d%16d%s\n",iSize,iTotal,iUnique,t) ;
  } 
}

