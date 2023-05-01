/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 */
#include <iostream>
#include <cstdlib>

using namespace std;

int N; //気温データ数
int P=3;      //*日以上borderを超えたら熱中症
int A[1009];//気温データ

int solve(int border){
  int currentLength=0;//連続日数
  for(int i=1;i<=N;i++){
    if(A[i]>border){
      currentLength++;
    }else{
      currentLength=0;
    }
    if(currentLength>=P){
      return false;
    }
  }
  return true;
}

int main(){
  N=20;     //20日間の気温データ
  for(int i=1;i<=N;i++){
    A[i]=rand()%50+1;//50度までの気温を乱数で入力
    printf("A[i]: %d\n",A[i]);
  }
  cout<<"適温としてありえる値のうち最小温度は";
  for(int i=1;i<=40;i++){
    if(solve(i)==true){
      cout<<i<<endl;
      break;
    }
  }
  return 0;
}
