/**
 * bash-5.1$ g++-10 2-3-1CPP.cpp && ./a.out
 * 数値を入れてください N=
 * 200 
 * A[i]=1 A[j]=2 A[k]=7
 * A[i]=1 A[j]=3 A[k]=6
 * A[i]=1 A[j]=4 A[k]=5
 * A[i]=2 A[j]=3 A[k]=5
 * 4
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
