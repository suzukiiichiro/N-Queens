#include <stdio.h>
#include <time.h>

#define MAX 27

long Total=1 ;        //合計解
long Unique=0;        //ユニーク解
enum { PLACE, REMOVE, DONE };
const int si=13;
const int msk=(1<<si)-1;
typedef struct{
	int BOUND1;
  int id;
  int aB[si];
  long lTotal;
  char step;
  int y;
  int startCol;
  int bm;
  int down;
  int right;
  int left;
}local; 
void place(local *l) {
  //int index=get_global_id(0);
  int aB[si];
  for (int i=0; i < si; i++)
    aB[i]=i;
  int BOUND1=l->BOUND1;
  //long lTotal=l->lTotal;
  long lTotal=0;
  //char step=l->step;
  char step=0;
  int y=0;
  int startCol =0;
  int bm=l->bm;
  int down=0;
  int right=0;
  int left=0;
  int i=1;
  printf("bound:%d:bm:%d:step:%c:donw:%d:right:%d:left:%d\n",BOUND1,bm,step,down,right,left);
  while (i!=0) {
  	i++;
    if (step==REMOVE) {
      if (y==startCol) {
        step=DONE;
        l->lTotal=lTotal;
        break;
      }
      bm=aB[--y];
    }
    int bit;
    bit=bm&-bm;
    //if(y==0){
    //  if(bm & (1<<BOUND1)){
    //    bit=1<<BOUND1;
        //aB[y]=bit;
    //  }else{
    //    step=DONE;
    //    break;
    //  }
   // }else{
    //  bit=bm&-bm;
      //aB[y]=bit;
    //}
    //
    //
    //
    //
    //qint  bit=bm & -bm;
          down ^= bit;
          right  ^= bit<<y;
          left  ^= bit<<(si-1-y);
    if (step==PLACE) {
      aB[y++]=bm;
      if (y==si) {
        lTotal += 1;
        step=REMOVE;
      } else {
        bm=msk & ~(down | (right>>y) | (left>>((si-1)-y)));
        if (bm==0)
          step=REMOVE;
//
      }
    } else {
      //if(y>0){
      bm ^= bit;
      //}
      if (bm==0)
        step=REMOVE;
      else
        step=PLACE;
    }
  }
  // Save kernel state for next round.
}
int main(void) {
  local l[si];              //構造体 local型 
  for(int i=0; i < si; i++){
		l[i].BOUND1=i;
    l[i].id=i;
    l[i].bm=(1<<si)-1;
    place(&l[i]);
  }
  for(int i=0; i < si; i++){
    printf("%d: %ld\n",l[i].id,l[i].lTotal);
  }
}
