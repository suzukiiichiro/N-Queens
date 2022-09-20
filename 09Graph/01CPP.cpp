/**
 * bash-5.1$ g++-10 01CPP.cpp && ./a.out
 * 数値を入れてください N=
 * 12
 * 数値を入れてください K=
 * 10
 * A[i]=1 A[j]=2 A[k]=7
 * A[i]=1 A[j]=3 A[k]=6
 * A[i]=1 A[j]=4 A[k]=5
 * A[i]=2 A[j]=3 A[k]=5
 * 4
 * bash-5.1$
 *
 */
#include <iostream>
using namespace std;

int N,K,A[59];
int cnt=0;

int main(){
  //12
  cout<<"数値を入れてください N= \n";
  cin>>N;
  //10
  cout<<"数値を入れてください K= \n";
  cin>>K;
  for(int i=1;i<=N;i++){
    A[i]=i;
  }
  for (int i=1;i<=N;i++){
    for (int j=i+1;j<=N;j++){
      for (int k=j+1;k<=N;k++){
        if (A[i]+A[j]+A[k]==K){
          cnt+=1;
          printf("A[i]=%d A[j]=%d A[k]=%d\n",A[i],A[j],A[k]);
        }
      }
    }
  }
  cout<<cnt<<endl;
  return 0;
}
