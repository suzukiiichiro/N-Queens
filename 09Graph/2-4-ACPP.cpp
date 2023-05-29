/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 * bash-5.1$ g++-10 2-4-ACPP.cpp && ./a.out
 * 5
 * 9
 * i:1 j:3 k:5
 * i:2 j:3 k:4
 * 求められる数は以下の通り数です
 * 2
 * bash-5.1$
 *
 */
#include <iostream>

using namespace std;

int main(){
  int N;//１からNまでの数
  int X;//重複無しで３つの数を選んだ求める合計
  int ans=0;//Xとなる通り数
  cin >>N>>X;
  for(int i=1;i<=N;i++){
    for(int j=i+1;j<=N;j++){
      int k=X-(i+j);
      if(j<k && k<=N){
        printf("i:%d j:%d k:%d\n",i,j,k);
        ans++;
      }
    }
  }
  cout<<"求められる数は以下の通り数です"<<endl;
  cout<<ans<<endl;
  return 0;
}
