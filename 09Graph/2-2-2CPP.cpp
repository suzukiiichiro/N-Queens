/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 */
#include <iostream>
using namespace std;

int main(){
  int N;
  int X;
  int ans=0;
  cout<<"Nの値を入力してください(例 10)"<<endl;
  cin >>N;
  cout<<"Xの値を入力してください(例 10)"<<endl;
  cin >>X;

  cout<<"1<=i<=j<=k<=N は以下のとおりです"<<endl;
  for(int i=1;i<=N;i++){
    for(int j=i+1;j<=N;j++){
      for(int k=j+1;k<=N;k++){
        if(i+j+k==X){
          printf("i:%d j:%d k:%d\n",i,j,k);
          ans+=1;
        }
      }
    }
  }
  cout<<ans<<endl;
  return 0;
}
