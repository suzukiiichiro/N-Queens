/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 */
#include <iostream>
using namespace std;

bool solve(int n){
  int cnt=0;
  for(int i=1;i<=n;i++){
    if(n%i==0){
      cnt++;
    }
  }
  if(cnt==8 && n%2==0){
    return true;
  }
  return false;
}
//
int main(){
  int N;
  int ans=0;
  cout << "105 は奇数であり正の約数を12個持ちます"<<endl;
  cout << "Nを入力してください(例 105)"<<endl;
  cin >>N;
  for(int i=1;i<=N;i++){
    if(solve(i)==true){
      ans++;
    }
  }
  cout<<ans<<endl;
  return 0;
}
