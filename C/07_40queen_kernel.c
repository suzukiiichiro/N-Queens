//  単体で動かすときは以下のコメントを外す
//#define GCC_STYLE
#ifndef OPENCL_STYLE
#include "stdio.h"
#include "stdint.h"
typedef int64_t qint;
int get_global_id(int dimension){ return 0;}
#define CL_KERNEL_KEYWORD
#define CL_GLOBAL_KEYWORD
#define CL_CONSTANT_KEYWORD
#define CL_PACKED_KEYWORD
#define SIZE 10
#else
//typedef long qint;
//typedef long int64_t;
//typedef ulong uint64_t;
  typedef long qint;
  typedef long int64_t;
  typedef ulong uint64_t;
  typedef ushort uint16_t;
#define CL_KERNEL_KEYWORD __kernel
#define CL_GLOBAL_KEYWORD __global
#define CL_CONSTANT_KEYWORD __constant
#define CL_PACKED_KEYWORD  __attribute__ ((packed))
#endif
#define MAX 27  
typedef struct{
    //  データを格納数る配列
    int array[MAX];
    //  現在の位置
    int current;
}STACK;
 
//  スタックの初期化
void init(STACK*);
//  値のプッシュ
int push(STACK*,int);
//  値のポップ
int pop(STACK*,int*);
//  スタックの初期化
void init(STACK* pStack)
{
    int i;
    for(i = 0; i < MAX; i++){
        pStack->array[i] = 0;
    }
    //  カレントの値を0に。
    pStack->current = 0;
}
//  値のプッシュ
int push(STACK* pStack,int value)
{
    if(pStack->current < MAX){
        //  まだデータが格納できるのなら、データを格納し、一つずらす。
        pStack->array[pStack->current] = value;
        pStack->current++;
        return 1;
    }
    //  データを格納しきれなかった
    return 0;
}
//  値のポップ
int pop(STACK* pStack,int* pValue)
{
    if(pStack->current > 0){
        //  まだデータが格納できるのなら、データを格納し、一つずらす。
        pStack->current--;
        *pValue = pStack->array[pStack->current];
        return *pValue;
    }
    return 0;
}
int leng(STACK* pStack)
{
    if(pStack->current > 0){
     return 1;
    }
    return 0;
}
struct CL_PACKED_KEYWORD queenState {
  qint BOUND1;
  int si;
  int id;
  qint aB[MAX];
  uint64_t lTotal; // Number of solutinos found so far.
  char step;
  char y;
  char startCol; // First column this individual computation was tasked with filling.
  qint bm;
  qint down;
  qint right;
  qint left;
};
void backtrack1(struct queenState *s){
  STACK Y;
  STACK LE;
  STACK DO;
  STACK RI;
  STACK BM;
  init(&Y);
  init(&LE);
  init(&DO);
  init(&RI);
  init(&BM);
  int y=s->y;
  int left=s->left;
  int down=s->down;
  int right=s->right;
  int sy=y;
  int sl=left;
  int sd=down;
  int sr=right; 
  int bend=0;
  int rflg=0;
  int bitmap;
  qint bit;
  int msk = (1 << s->si) - 1;
  uint16_t i = 1;
  while (i != 0) {
  	i++;
    //start:
    //printf("rflg:%d\n",rflg);
    if(rflg==0){
      //printf("#######method_start#####\n");
      //printf("y:%d:left:%d:down:%d:right:%d\n",y,left,down,right);
      bit;
      //printf("int  bitmap, bit;\n");

      bitmap = msk & ~(left | down | right);
      //printf("bitmap = MASK & ~(left | down | right);\n");
    }

    if (y == s->si && rflg==0) {
      //printf("if (y == SIZEE) {\n");
      if (bitmap) {
        //printf("if (bitmap) {\n");
        s->aB[y] = bitmap;
        //printf("BOARD[y] = bitmap;\n");
        s->lTotal += 8;
        //printf("COUNT8++;\n");
        //Display();
      }
      //printf("}\n");
    } else {
      if (y < s->BOUND1 && rflg==0) {   /* 斜軸反転解の排除 */
        //printf("} else { #y !=SIZEE\n");
        //printf("if (y < BOUND1) { \n");
        bitmap |= 2;
        //printf("bitmap |= 2;\n");
        bitmap ^= 2;
        //printf("bitmap ^= 2;\n");
        //printf("}\n");
      }
      while (bitmap||rflg==1) {
        //printf("while (bitmap) {\n");
        if(rflg==0){
          bitmap ^= s->aB[y] = bit = -bitmap & bitmap;
          //printf("bitmap ^= BOARD[y] = bit = -bitmap & bitmap;\n");
          //printf("Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);\n");
          //printf("###rec:y+1:%d:(left|bit)<<1:%d:down|bit:%d:(right|bit)>>1:%d\n",y+1,(left|bit)<<1,down|bit,(right|bit)>>1);
          //printf("#bit:%d:bitmap:%d:BOUND1:%d\n",bit,bitmap,BOUND1);
          //      Backtrack1(y+1, (left | bit)<<1, down | bit, (right | bit)>>1);
          push(&Y,y); 
          push(&LE,left);
          push(&DO,down);
          push(&RI,right);
          push(&BM,bitmap);
          y=y+1;
          left=(left|bit)<<1;
          down=(down|bit);
          right=(right|bit)>>1; 
          //goto start;
          bend=1;
          break;
        }
        //ret:
        if(rflg==1){
          if(leng(&Y)!=0){
            y=pop(&Y,&y);
            left=pop(&LE,&left);
            down=pop(&DO,&down);
            right=pop(&RI,&right);
            bitmap=pop(&BM,&bitmap);
            rflg=0;
            //printf("#after_backtrack1\n");
            //printf("#after:bit:%d:bitmap:%d:BOUND1:%d\n",bit,bitmap,BOUND1);
            //for (int i=0; i<SIZE; i++) {
            //  printf("after_BOARD[%d]:%d\n",i,BOARD[i]);
            //}
          }
        }
      }
      if(bend==1 && rflg==0){
        bend=0;
        continue;
      }
      //printf("}#while(bitmap)end#\n");
      //printf("#pop#y:%d:left:%d:down:%d:right:%d\n",y,left,down,right);
      //printf("#pop#bit:%d:bitmap:%d:BOUND1:%d\n",bit,bitmap,BOUND1);
      //for (int i=0; i<SIZE; i++) {
      //  printf("BOARD[%d]:%d\n",i,BOARD[i]);
      //}

    }
    //printf("##methodend}\n");
    if(y==sy && left==sl && down == sd && right==sr){
      s->step=2;
      break;
    }else{
      //goto ret;
      rflg=1;
    }
  }
  s->y=y;
  s->right=right;
  s->down=down;
  s->left=left;
}
void backtrack2(struct queenState *s){

  int msk = (1 << s->si) - 1;
  qint bit;
  uint16_t i = 1;
  //long i=0;
  //while (i <300000)
  while (i != 0) {
  	i++;
    if(s->step==1){
      if(s->y<=s->startCol){
        s->step=2;
        break;
      }
      s->bm=s->aB[--s->y];
    }
    if(s->y==0){
      if(s->bm & (1<<s->BOUND1)){
        bit=1<<s->BOUND1;
        s->aB[s->y]=bit;
      }else{
        s->step=2;
        break;
      }
    }else{
      bit=s->bm&-s->bm;
      s->aB[s->y]=bit;
    }
    s->down  ^= bit;
    s->right ^= bit<<s->y;
    s->left  ^= bit<<(s->si-1-s->y);
    if(s->step==0){
      s->aB[s->y++]=s->bm;
      if(s->y==s->si){
        s->lTotal += 1;
        s->step=1;
      }else{
        s->bm=msk&~(s->down|(s->right>>s->y)|(s->left>>((s->si-1)-s->y)));
        if(s->bm==0)
          s->step=1;
      }
    }else{
      s->bm ^= bit;
      if(s->bm==0)
        s->step=1;
      else
        s->step=0;
    }
  }
}
CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState *state){
  int index = get_global_id(0);
	struct queenState s ;
  s.BOUND1=state[index].BOUND1;
  s.si= state[index].si;
  s.id= state[index].id;
  //int aB[MAX];
  for (int i = 0; i < s.si; i++)
    s.aB[i] = state[index].aB[i];
  s.lTotal = state[index].lTotal;
  s.step      = state[index].step;
  s.y       = state[index].y;
  s.startCol  = state[index].startCol;
  s.bm     = state[index].bm;
  s.down     = state[index].down;
  s.right      = state[index].right;
  s.left      = state[index].left;
//  printf("bound:%d:startCol:%d:ltotal:%ld:step:%d:y:%d:bm:%d:down:%d:right:%d:left:%d\n", BOUND1,startCol,lTotal,step,y,bm,down,right,left);
  if(s.step !=2){//step がDONEの時は処理してもしょうがないので抜ける
    if(s.BOUND1==0){
      backtrack1(&s);
    }else{
      backtrack2(&s);
    } 
  }
  state[index].BOUND1   =s.BOUND1;
  state[index].si      = s.si;
  state[index].id      = s.id;
  for (int i = 0; i < s.si; i++)
    state[index].aB[i] = s.aB[i];
  state[index].lTotal = s.lTotal;
  state[index].step      = s.step;
  state[index].y       = s.y;
  state[index].startCol  = s.startCol;
  state[index].bm      = s.bm;
  state[index].down      = s.down;
  state[index].right       = s.right;
  state[index].left       = s.left;
}
#ifdef GCC_STYLE
int main(){
  int si=10; 
  struct queenState l[SIZE*SIZE*SIZE];
  long gTotal=0;
  for (int i=0;i<SIZE;i++){
    for(int j=0;j<si;j++){
      for(int k=0;k<si;k++){
    l[i*si*si+j*si+k].BOUND1=i;
    l[i*si*si+j*si+k].BOUND2=j;
    l[i*si*si+j*si+k].BOUND3=k;
    l[i*si*si+j*si+k].si=si;
    for (int m=0;m< si;m++){
      l[i*si*si+j*si+k].aB[m]=m;
    }
    l[i*si*si+j*si+k].step=0;
    l[i*si*si+j*si+k].y=0;
    l[i*si*si+j*si+k].startCol=3;
//    l[i].msk=(1<<SIZE)-1;
    l[i*si*si+j*si+k].bm=(1<<SIZE)-1;
    l[i*si*si+j*si+k].down=0;
    l[i*si*si+j*si+k].right=0;
    l[i*si*si+j*si+k].left=0;
    l[i*si*si+j*si+k].lTotal=0;
    place(&l[i*si*si+j*si+k]);
    gTotal+=l[i*si*si+j*si+k].lTotal;
    printf("%ld\n", l[i*si*si+j*si+k].lTotal);
      }
    }
  }
  printf("%ld\n", gTotal);
  return 0;
}
#endif
