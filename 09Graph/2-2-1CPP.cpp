/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 */
#include <iostream>
using namespace std;

bool solve(int n){
  int cnt=0;
  if(n%2==0){
    return true;
  }
  return false;
}
//
int main(){
  int N;
  int ans=0;
  cout << "偶数を数えます"<<endl;
  cout << "Nを入力してください(例 10)"<<endl;
  cin >>N;
  for(int i=1;i<=N;i++){//1から数え始める
    if(solve(i)==true){
      ans+=1;
    }
  }
  cout<<ans<<endl;
  return 0;
}
