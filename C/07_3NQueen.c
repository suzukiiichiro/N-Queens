#include <stdio.h>
#include <time.h>
#define MAXSIZE 12
// 日本語

int T=1 ;
int U;
int fa[MAXSIZE];
int fb[MAXSIZE];
int fc[MAXSIZE];
int pos[MAXSIZE];
void TimeFormat(clock_t utime, char *form)
{
    int  dd, hh, mm;
    float ftime, ss;

    ftime = (float)utime / CLOCKS_PER_SEC;

    mm = (int)ftime / 60;
    ss = ftime - (float)(mm * 60);
    dd = mm / (24 * 60);
    mm = mm % (24 * 60);
    hh = mm / 60;
    mm = mm % 60;

    if (dd) sprintf(form, "%4d %02d:%02d:%05.2f", dd, hh, mm, ss);
    else if (hh) sprintf(form, "     %2d:%02d:%05.2f", hh, mm, ss);
    else if (mm) sprintf(form, "        %2d:%05.2f", mm, ss);
    else sprintf(form, "           %5.2f", ss);
}
void NQueen3(int i,int s){
  int j; //# s:size
  for(j=0;j<s;j++){
    if(fa[j]==0 && fb[i+j]==0 && fc[i-j+s-1]==0){
      pos[i]=j ;
      if(i==s-1){
        T++;
      }else{
        fa[j]=1;
        fb[i+j]=1; 
        fc[i-j+s-1]=1; 
        NQueen3(i+1,s); 
        fa[j]=0;           
        fb[i+j]=0;   
        fc[i-j+s-1]=0; 
      }          
    }
  }  
}

int main(void)
{
  //# m: max mi:min s:size st:starttime t:time
  int m=MAXSIZE; int mi=2; int s=mi; clock_t st; char t[20];
  printf("%s"," N:        Total       Unique        hh:mm:ss.");
  printf("\n");
  for(s=mi;s<=m;s++){
    T=0; U=0; st=clock();
    NQueen3(0,s);
    TimeFormat(clock() - st, t);
    printf("%2d:%13d%13d%s\n",s,T,U,t) ;
  } 
}

