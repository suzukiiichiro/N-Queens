#include <stdio.h>
#include <stdbool.h>
/**
 *
 */
void swap(int *pa,int *pb)
{
  int tmp;
  tmp=*pa;
  *pa=*pb;
  *pb=tmp;
}
/**
 *
 */
void reverse(size_t first,size_t last,int v[])
{
  while(first != last && first != --last){
    swap(&v[first],&v[last]);
    first++;
  }
}
/**
 *
 */
bool next_permutation(size_t first,size_t last,int v[])
{
  size_t i,j,k;
  if(first == last){
    return false;
  }
  if(first + 1 == last){
    return false;
  }
  i=last - 1;
  while(true){
    j=i--;
    if(v[i] < v[j]){
      k=last;
      while(!(v[i] < v[--k])){
      }
      swap(&v[i],&v[k]);
      reverse(j,last,v);
      return true;
    }
    if(i == first){
      reverse(first,last,v);
      return false;
    }
  }
}
/**
 *
 */
int main(void)
{
  int v[] ={1,2,3,4,5};
  size_t N=sizeof(v) / sizeof(v[0]);
  int COUNT=1;
  do{
    printf("%d: ",COUNT++);
    for(int i=0; i < N; i++){
      printf("%d ",v[i]);
    }
    printf("\n");
  }while(next_permutation(0,N,v));
  return 0;
}
