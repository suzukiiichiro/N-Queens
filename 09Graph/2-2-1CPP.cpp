/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 */
#include <iostream>
using namespace std;

bool solve(int n){
  int cnt=0;
  if(n%2==1){
    //printf("奇数：%d\n",n);
    for(int i=1;i<=n;i++){
      if(i%2==0){
        //printf("正の約数 %d: %d\n",n,i);
        cnt++;
        if(cnt==8){
          //printf("正の約数を８個持つ %d:",i);
          return true;
        }
      }
    }
  }
  return false;
}
//
int main(){
  int N;
  int ans=0;

  cout << "105 は奇数であり正の約数を12個持ちます"<<endl;
  cout << "200以下の奇数のうち正の約数を８個持つ数はいくつか"<<endl;

  cout << "Nを入力してください(例200)"<<endl;
  cin >>N;

  for(int i=1;i<=N;i++){
    if(solve(i)==true){
      ans++;
    }
  }
  cout<<"1<=N<=200以下の奇数で、";
  cout<<"正の約数をちょうど８個持つ数の";
  cout<<"個数は以下のとおりです。"<<endl;
  cout<<ans<<endl;
  return 0;
}
