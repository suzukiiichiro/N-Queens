/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 */
#include <iostream>
#include <cstdlib>

using namespace std;

int N=20;
int P=3;
int A[1009];
int ans;

bool solve(int border){
  int currentLength=0;
  for(int i=1;i<=N;i++){
    if(A[i]>border){
      currentLength++;
    }else{
      currentLength=0;  
    }
    if(currentLength>=3){
      return false;
    }
  }
  return true;
}
//
int main(){
  for(int i=0;i<N;i++){
    A[i]=rand()%50+1;//50度までの気温を乱数で入力
    printf("A[i]: %d\n",A[i]);
  }
  for(int i=1;i<=N;i++){
    if(solve(A[i])==false){
      ans=min(ans,A[i]);
    }
  }
  cout<<ans<<endl;
  return 0;
}
