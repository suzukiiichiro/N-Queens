/**
 * bash-5.1$ g++-10 2-3-1CPP.cpp && ./a.out
 * 数値を入れてください N=
 * 200 
 * ans:105:A[0]=1,A[1]=3,A[2]=5,A[3]=7,A[4]=15,A[5]=21,A[6]=35,A[7]=105,
 * ans:135:A[0]=1,A[1]=3,A[2]=5,A[3]=9,A[4]=15,A[5]=27,A[6]=45,A[7]=135,
 * ans:165:A[0]=1,A[1]=3,A[2]=5,A[3]=11,A[4]=15,A[5]=33,A[6]=55,A[7]=165,
 * ans:189:A[0]=1,A[1]=3,A[2]=7,A[3]=9,A[4]=21,A[5]=27,A[6]=63,A[7]=189,
 * ans:195:A[0]=1,A[1]=3,A[2]=5,A[3]=13,A[4]=15,A[5]=39,A[6]=65,A[7]=195,
 * 5
 * bash-5.1$
 *
 *105は奇数であり正の約数を 8 個持ちます。さて、1 以上 N 以下の奇数のうち、正の約数をちょうど 8 個持つ数の個数はいくつでしょうか？
制約：1≤N≤200
 */
#include <iostream>
using namespace std;

int A[200];
bool solve(int n) {
    int cnt = 0;
    for (int i = 1; i <= n; i++) {
        if (n % i == 0){ 
          A[cnt]=i;
          cnt += 1;
        }
    }
    if (cnt == 8 && n % 2 == 1){//先に奇数以外弾くのは枝刈り 
      printf("ans:%d:",n); 
      for(int i=0; i<cnt; i++){
       printf("A[%d]=%d,",i,A[i]);
      }
      printf("\n");
      return true;
    }
    return false;
}

int main() {
    int N, ans = 0;
    cout<<"数値を入れてください N= \n";
    cin >> N;
    for (int i = 1; i <= N; i++) {
        if (solve(i) == true) ans += 1;
    }
    cout << ans << endl;
    return 0;
}
