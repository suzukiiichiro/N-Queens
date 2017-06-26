//お姉さんロボ
//
#include <stdio.h>
#include <time.h>

#define SIZE 8 
int aGrid[112][2];//TODO 配列のサイズを動的にする

//２分木のグラフを作る
void GridGraph(){
  int n=SIZE;
  int id=0;
  aGrid[0][0]=0;
  aGrid[0][1]=1;
  int m=0;
  for (int i=1;i<=2*(n-1);i++){
    if (i<n){
      m=2*i; 
    }else{
      m=2*(2*(n-1)-i+1); 
    }
    for (int j=1; j<=m;j++){
      if(j==1){
        aGrid[id][0]++;
        aGrid[id][1]++;
      }else if((i<n && j%2==0)||(i >=n && j%2==1)){
        aGrid[id][1]++; 
      }else{
        aGrid[id][0]++; 
      }
      id++;
      aGrid[id][0]=aGrid[id-1][0]; 
      aGrid[id][1]=aGrid[id-1][1]; 
    } 
  }

}

int main(void) {
//まず２分木のグラフを作る
  GridGraph(); 
  for(int i=0;i<=111;++i){
    printf("%d:%d\n", aGrid[i][0],aGrid[i][1]);
  }


}
