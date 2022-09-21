/**
 * bash-5.1$ g++-10 **CPP.cpp && ./a.out
 *
 */
#include <iostream>
#include <cstdlib>

using namespace std;

int N; //気温データ数
int A[1009];//気温データ

int solve(int tmp){
  int border=40;//４０度以上を暑いと感じる
  int P=3;//３日以上borderを超えたら熱中症
  int currentLength=0;//連続日数
  for(int i=1;i<=N;i++){
    if(tmp>border){
      currentLength++;
    }else{
      currentLength=0;
    }
    if(currentLength>=P){
      return true;
    }
  }
  return false;
}

int main(){
  N=1000; //1000日間の気温データ
  //printf("N:%d P:%d\n",N,P);

  for(int i=1;i<=N;i++){
    A[i]=rand()%50+1;//40度までの気温を乱数で入力
    printf("A[i]: %d\n",A[i]);
  }
  cout<<"連続で暑いと感じた最小連続日時は";
  for(int i=0;i<=N;i++){
    if(solve(A[i])==true){
      cout<<i<<"です。"<<endl;
      break;
    }
  }
  return 0;
}
