#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include<iostream> 
#include<sstream>//int型の変数をstring型に変換するために使用 
#include<string>//stringを使用するため 
#include<vector>//リスト構造を持つ配列vectorを使用するため 
#define N 8 
#define MAX 10000
//#include <windows.h> 


/**
 * *@param Q[N/2] 左上部分にあたる1/4の部分解の駒の位置を保存している
 * *@param A[N] TypeAの駒の配置を保存している
 * *@param B[N] TypeBの駒の配置を保存している
 * *@param C[N] TypeCの駒の配置を保存している
 * */
void init(int Q[N/2],int A[N],int B[N],int C[N]){

  for(int i=0;i<N/2;i++){
    Q[i]=N+1;
  }

  for(int i=0;i<N;i++){
    A[i]=N+1;
    B[i]=N+1;
    C[i]=N+1;
  }

}





/*
* 1/4の部分解作成メソッド
*/
/**
*@param Q[N/2] 左上部分にあたる1/4の部分解の駒の位置を保存している
*@param A[N] TypeAの駒の配置を保存している
*@param B[N] TypeBの駒の配置を保存している
*@param C[N] TypeCの駒の配置を保存している
*@param qcount 最初に配置した駒の数を保存している
*/
void partget(int Q[N/2],int A[N],int B[N],int C[N],int &qcount){

  int d=0;//駒配置の数の変数 
  std::vector<int> vector;
  //配置する個数を決定する。 
  do{
    d = rand()%(N/2);
  }while((N/4) > d); 
  
  qcount =d;
  //1/4部分解作成(左上部分の作成)
  //vectorに0～N/2までの数を代入 
  for(int i=0;i<N/2;i++){
    vector.push_back(i);
  }

  //vector内でランダムに並び替え 
  int size = vector.size(); 
  for(int j=0;j<size;j++){
    int w = vector[j];
    int r = rand()%(size); 
    vector[j] = vector[r]; 
    vector[r] = w;
  }

  //vector全体から駒の数だけ残し、あとはからにする 
  size = vector.size() - d;
  for(int i=0;i<size;i++){
    vector.pop_back();
  }
  //配置なしのN+1を入れる 
  for(int i=0;i<size;i++){
    vector.push_back(N+1);
  }

  //vector内でランダムに並び替え 
  size = vector.size();
  for(int j=0;j<size;j++){
    int w = vector[j];
    int r = rand()%(size); 
    vector[j] = vector[r]; 
    vector[r] = w;
  }

  //vector内の数をQ[N/2]に代入することで駒の配置を実現
  //左上の1/4の部分解の完成 
  for(int c=0;c<N/2;c++){
    Q[c]=vector[c];
    A[c]=vector[c];
    B[c]=vector[c];
    C[c]=vector[c];
    printf("A:%d B:%d C:%d Q:%d\n",A[c],B[c],C[c],Q[c]);
  }

}

/*
* TypeAの1/2の部分解作成メソッド
*/
/**
*@param A[N] TypeCの駒の配置を保存している
*@param qcount 最初に配置した駒の数を保存している
*/
void partgetA(int A[N],int &qcount){ 
  std::vector<int> vector;
  //1/4部分解作成(右下部分の作成)
  //vectorにN/2～Nまでの数を代入 
  for(int i=N/2;i<N;i++){
    vector.push_back(i);
  }

  //vector内でランダムに並び替え 
  int size = vector.size(); 
  for(int j=0;j<size;j++){
    int w = vector[j];
    int r = rand()%(size); 
    vector[j] = vector[r]; 
    vector[r] = w;
  }

  //vector全体から駒の数だけ残し、あとはからにする 
  size = vector.size() - qcount;
  for(int i=0;i<size;i++){
    vector.pop_back();
  }
  //配置なしのN+1を入れる 
  for(int i=0;i<size;i++){
    vector.push_back(N+1);
  }
  //vector内でランダムに並び替え 
  size = vector.size();
  for(int j=0;j<size;j++){
    int w = vector[j];
    int r = rand()%(size); 
    vector[j] = vector[r]; 
    vector[r] = w;
  }

  //右下の1/4の部分解の完成 
  for(int c=0;c<N/2;c++){
    A[c+(N/2)]=vector[c];
  }
}

/*
* TypeBの1/2の部分解作成メソッド
*/
/**
*@param Q[N/2] 左上部分にあたる1/4の部分解の駒の位置を保存している
*@param B[N] TypeBの駒の配置を保存している
*/
void partgetB(int Q[N/2],int B[N]){

  int newb1[N/2];//90°回転した駒の配置を保存 
  int newb2[N/2];//180°回転した駒の配置を保存

  //1/4部分解作成(右下部分の作成)

  //90°回転させる 
  for(int j=0;j<N/2;j++){
    for(int k=0;k<N/2;k++){
      if(Q[k]==j){
        newb1[j]=((N/2)-1)-k; break;
      }
      else{
        newb1[j]=N+1;

      }
    }
  }
  //90°回転させる 
  for(int e=0;e<N/2;e++){
    for(int v=0;v<N/2;v++){
      if(newb1[v]==e){
        newb2[e]=(N/2-1)-v; break;
      }
      else{
        newb2[e]=N+1;
      }
    }
  }

  for(int i=0;i<N/2;i++){
    if(newb2[i] != N+1){
      B[i+(N/2)]=newb2[i]+(N/2);
    }
  }
}

/*
* TypeCの1/2の部分解作成メソッド
*/
/**
*@param Q[N/2] 左上部分にあたる1/4の部分解の駒の位置を保存している
*@param C[N] TypeCの駒の配置を保存している
*/
void partgetC(int Q[N/2],int C[N]){

  int newc1[N/2];//90°回転した駒の配置を保存 
  int newc2[N/2];//180°回転した駒の配置を保存 
  int newc3[N/2];//270°回転した駒の配置を保存

  //1/4部分解作成(右上部分の作成)

  //90°回転させる 
  for(int j=0;j<N/2;j++){
    for(int k=0;k<N/2;k++){
      if(Q[k]==j){
        newc1[j]=((N/2)-1)-k;
        break;
      }
      else{
        newc1[j]=N+1;
      }
    }
  }
  //90°回転させる 
  for(int j=0;j<N/2;j++){
    for(int k=0;k<N/2;k++){
      if(newc1[k]==j){
        newc2[j]=((N/2)-1)-k; 
        break;
      }
      else{
        newc2[j]=N+1;
      }     
    }
  }
  //90°回転させる 
  for(int j=0;j<N/2;j++){
    for(int k=0;k<N/2;k++){
      if(newc2[k]==j){
        newc3[j]=((N/2)-1)-k; 
        break;
      }
      else{
        newc3[j]=N+1;
      }   
    }
  }
  for(int i=0;i<N/2;i++){
    if(newc1[i] != N+1){
      C[i]=newc1[i]+(N/2);
    }
  }
  for(int i=0;i<N/2;i++){
    if(newc2[i] != N+1){
      C[i+(N/2)] = newc3[i];
    }
  }
}
/**
*@param A[N] TypeAの駒の配置を保存している
*@param judgeA TypeAの判定を行うかの判定用変数が格納されている
*/
void checkA(int A[N],int &judgeA){ 
  int x=0;
  int p1[2*N-1];//右斜め上の配列 
  int q1[2*N-1];//右斜め下の配列 
  int sum=0;

  //配列の初期化
  for(int t=0;t<2*N-1;t++){ 
    p1[t]=0;
    q1[t]=0;
  }

  //a[N]の解判定 
  for(int j=0;j<N;j++){
    x=A[j];//コマの位置情報をxに代入 
    if(x != N+1){
    //右上斜め判定 
      if(p1[j+x]==0){
        p1[j+x]=1;
 
      }
      else{
        sum+=1;
      }
      //右下斜め判定 
      if(q1[j-x+(N-1)]==0){
        q1[j-x+(N-1)]=1;
      }
      else{
        sum+=1;
      }
    }
  }
 
  if(sum==0){
    judgeA=1;
  }
  sum=0;
}
 

/**
*@param B[N] TypeBの駒の配置を保存している
*@param judgeB TypeBの判定を行うかの判定用変数が格納されている
*/
void checkB(int B[N],int &judgeB){ 
  int x=0;
  int p1[2*N-1];//右斜め上の配列 
  int q1[2*N-1];//右斜め下の配列 
  int sum=0;

  //配列の初期化
  for(int t=0;t<2*N-1;t++){ 
    p1[t]=0;
    q1[t]=0;
  }

  //a[N]の解判定 
  for(int j=0;j<N;j++){
    x=B[j];//コマの位置情報をxに代入 
    if(x != N+1){
    //右上斜め判定 
      if(p1[j+x]==0){
        p1[j+x]=1;
      }
      else{
        sum+=1;
      }
      //右下斜め判定 
      if(q1[j-x+(N-1)]==0){
        q1[j-x+(N-1)]=1;
 
      }
      else{
        sum+=1;
      }
    }
  }
  if(sum==0){
    judgeB=1;
  }
  sum=0;
}
/**
*@param C[N] TypeCの駒の配置を保存している
*@param judgeC TypeCの判定を行うかの判定用変数が格納されている
*/
void checkC(int C[N],int &judgeC){ 
  int x=0;
  int p1[2*N-1];//右斜め上の配列 
  int q1[2*N-1];//右斜め下の配列 
  int r1[N];
  int sum=0;

  //配列の初期化
  for(int t=0;t<2*N-1;t++){ 
    p1[t]=0;
    q1[t]=0;
  }
  for(int s=0;s<N;s++){
    r1[s]=0;
  }


  //a[N]の解判定 
  for(int j=0;j<N;j++){
    x=C[j];//コマの位置情報をxに代入 
    if(x != N+1){
      //右上斜め判定 
      if(p1[j+x]==0){
        p1[j+x]=1;
      }
      else{
        sum+=1;
      }
      //右下斜め判定 
      if(q1[j-x+(N-1)]==0){
        q1[j-x+(N-1)]=1;
      }
      else{
        sum+=1;
      }
      //縦判定 
      if(r1[x]==0){
        r1[x]=1;
 
      }
      else{
        sum +=1;
      }

    }
  }

  if(sum==0){
    judgeC=1;
  }
  sum=0;
}
/*
* TypeAの残りの駒の配置をすべておくメソッド
*/
/**
*@param A[N] TypeAの駒の配置を保存している
*/
void allputA(int A[N]){
  int r[N];//縦判定用配列
  int p[2*N-1];//右斜め上判定用配列 
  int q[2*N-1];//右斜め下判定用配列
  int putcount=0;//配置できる部分の数を入れる変数
  int putprace=0;//配置できる部分を保存するための変数
  //判定用配列の初期化 
  for(int x=0;x<N;x++){
    r[x]=0;
  }
  for(int y=0;y<2*N-1;y++){ 
    p[y]=0;
    q[y]=0;
  }
  //すでにある駒の利き筋を保存 
  for(int s=0;s<N;s++){
    int t=A[s];//コマの位置情報をtに代入
    if(t != N+1){
      //右上斜め判定 
      p[s+t]=1;
      //右下斜め判定 
      q[s-t+(N-1)]=1;
      //縦判定 
      r[t]=1;
    }
  }
  //すべての駒の配置を確認するまで続ける 
  for(int e=0;e<N;e++){
  //すべての駒の配置をうめる 
    for(int i=0;i<N;i++){
    //駒がおいてないところがある場合 
      if(A[i]==N+1){
      //右上部分にあたる時 
        if(i<N/2){
        //駒がおけるかどうか確認する 
          for(int j=N/2;j<N;j++){
          //配置場所が利き筋でないかどうか判定 
            if(r[j]==0){
              if(p[i+j]==0){
                if(q[i-j+(N-1)]==0){
                //すべて0なら配置カウント+1
                  putcount +=1; 
                  putprace=j;
 
                }
              }
            }
          }
          //配置カウントが1なら駒の配置 
          if(putcount==1){
            A[i] = putprace; 
            r[putprace]=1; 
            p[i+putprace]=1;
            q[i-putprace+(N-1)]=1;
          }
        }
        //左下部分にあたる時 
        else{
          //駒がおけるかどうか確認する 
          for(int g=0;g<N/2;g++){
            //配置場所が利き筋でないかどうか判定 
            if(r[g]==0){
              r[g]=1;
              if(p[i+g]==0){
                p[i+g]=1;
                if(q[i-g+(N-1)]==0){
                //すべて0なら配置カウント+1し、場所を保存

                  putcount +=1; 
                  putprace=g;
                }
                else{
                  r[g]=0;
                  p[i+g]=0;
                }
              }
              else{
                r[g]=0;
              }
            }
          }
          //配置カウントが1なら駒の配置 
          if(putcount==1){
            A[i] = putprace; 
            r[putprace]=1; 
            p[i+putprace]=1;
            q[i-putprace+(N-1)]=1;

          }           
        }
      }
      //カウントと場所を初期化する 
      putcount=0;
      putprace=0;
    }
  }
}  

/*
* TypeBの残りの駒の配置をすべておくメソッド
*/

/**
*@param B[N] TypeBの駒の配置を保存している
*/
void allputB(int B[N]){

  int r[N];//縦判定用配列
  int p[2*N-1];//右斜め上判定用配列 
  int q[2*N-1];//右斜め下判定用配列
  int putcount=0;//配置できる部分の数を入れる変数
  int putprace=0;//配置できる部分を保存するための変数

  //判定用配列の初期化 
  for(int x=0;x<N;x++){
    r[x]=0;
  }
  for(int y=0;y<2*N-1;y++){ 
    p[y]=0;
    q[y]=0;
  }

  //すでにある駒の利き筋を保存 
  for(int s=0;s<N;s++){
    int t=B[s];//コマの位置情報をtに代入 
    if(t != N+1){
    //右上斜め判定 
      p[s+t]=1;
    //右下斜め判定 
      q[s-t+(N-1)]=1;
    //縦判定 
      r[t]=1;
    }
  }

  //すべての駒の配置を確認するまで続ける 
  for(int e=0;e<N;e++){
    //すべての駒の配置をうめる 
    for(int i=0;i<N;i++){

      //駒がおいてないところがある場合 
      if(B[i]==N+1){

        //右上部分にあたる時 
        if(i<N/2){
          //駒がおけるかどうか確認する



          for(int j=N/2;j<N;j++){

            //配置場所が利き筋でないかどうか判定
            if(r[j]==0){
              if(p[i+j]==0){
                if(q[i-j+(N-1)]==0){
                  //すべて0なら配置カウント+1

                  putcount +=1;
                  putprace=j;
                }
              }
            }
          }
          //配置カウントが1なら駒の配置
          if(putcount==1){
            B[i] = putprace;
            r[putprace]=1;
            p[i+putprace]=1;
            q[i-putprace+(N-1)]=1;
          }
        } 

        //左下部分にあたる時
        else{
          //駒がおけるかどうか確認する
          for(int g=0;g<N/2;g++){
            //配置場所が利き筋でないかどうか判定
            if(r[g]==0){
              r[g]=1;
              if(p[i+g]==0){
                p[i+g]=1;
                if(q[i-g+(N-1)]==0){
                  //すべて0なら配置カウント+1し、場所を保存
                  putcount +=1;
                  putprace=g;
                }
                else{
                  r[g]=0;
                  p[i+g]=0;
                }
              }
              else{
                r[g]=0;
              } 

            }
          }
          //配置カウントが1なら駒の配置
          if(putcount==1){
            B[i] = putprace;
            r[putprace]=1;
            p[i+putprace]=1;
            q[i-putprace+(N-1)]=1;

          }
        }
      }
      //カウントと場所を初期化する
      putcount=0;
      putprace=0;
    }
  }
} 
/*
* TypeCの残りの駒の配置をすべておくメソッド
*/
/**
*@param C[N] TypeCの駒の配置を保存している
*/
void allputC(int C[N]){
  int r[N];//縦判定用配列
  int p[2*N-1];//右斜め上判定用配列
  int q[2*N-1];//右斜め下判定用配列
  int putcount=0;//配置できる部分の数を入れる変数
  int putprace=0;//配置できる部分を保存するための変数
  //判定用配列の初期化
  for(int x=0;x<N;x++){
    r[x]=0;
  }
  for(int y=0;y<2*N-1;y++){
    p[y]=0;

    q[y]=0;
  }
  //すでにある駒の利き筋を保存
  for(int s=0;s<N;s++){
    int t=C[s];//コマの位置情報をtに代入
    if(t != N+1){
      //右上斜め判定
      p[s+t]=1;
      //右下斜め判定
      q[s-t+(N-1)]=1;
      //縦判定
      r[t]=1;
    }
  }
  //すべての駒の配置を確認するまで続ける
  for(int e=0;e<N;e++){
    //すべての駒の配置をうめる
    for(int i=0;i<N;i++){ 

      //駒がおいてないところがある場合
      if(C[i]==N+1){
        //右上部分にあたる時
        if(i<N/2){
          //駒がおけるかどうか確認する
          for(int j=N/2;j<N;j++){
            //配置場所が利き筋でないかどうか判定
            if(r[j]==0){
              if(p[i+j]==0){
                if(q[i-j+(N-1)]==0){
                  //すべて0なら配置カウント+1

                  putcount +=1;
                  putprace=j;
                }
              }
            }
          }
          //配置カウントが1なら駒の配置
          if(putcount==1){
            C[i] = putprace;
            r[putprace]=1;
            p[i+putprace]=1;
            q[i-putprace+(N-1)]=1; 

          }
        }
        //左下部分にあたる時
        else{
          //駒がおけるかどうか確認する
          for(int g=0;g<N/2;g++){
            //配置場所が利き筋でないかどうか判定
            if(r[g]==0){
              r[g]=1;
              if(p[i+g]==0){
                p[i+g]=1;
                if(q[i-g+(N-1)]==0){
                  //すべて0なら配置カウント+1し、場所を保存

                  putcount +=1;
                  putprace=g;
                }
                else{
                  r[g]=0;
                  p[i+g]=0;
                }
              }
              else{
                r[g]=0;
              }
            }
          }
          //配置カウントが1なら駒の配置
          if(putcount==1){
            C[i] = putprace;

            r[putprace]=1;
            p[i+putprace]=1;
            q[i-putprace+(N-1)]=1;

          }
        }
      }
      //カウントと場所を初期化する
      putcount=0;
      putprace=0;
    }
  }
} 
 
/**
*@param A[N] TypeAの駒の配置を保存している
*@param vector 解となった駒の配置情報を保存する
*/
void fcheckA(int A[N],std::vector<std::string>& vector){
  int x=0;
  int p[2*N-1];//右斜め上の配列
  int q[2*N-1];//右斜め下の配列
  int sum=0;
  int size=0;
  int no=0;
  int newa[N];
  int newb[N];
  int newc[N];
  std::ostringstream l;
  std::ostringstream m;
  std::ostringstream m1;
  std::ostringstream n;
  std::ostringstream h1;
  std::ostringstream h2;
  std::ostringstream o1;
  std::ostringstream o2;
  std::ostringstream z; 

  //配列の初期化
  for(int t=0;t<2*N-1;t++){
    p[t]=0;
    q[t]=0;
  }
  //配置なしの場合を確認
  for(int i=0;i<N;i++){
    if(A[i]==N+1){
      no=1;
      break;
    }
  } 

  if(no == 0){
  //B[N]の解判定
    for(int j=0;j<N;j++){
      x=A[j];//コマの位置情報をxに代入
      //右上斜め判定
      if(p[j+x]==0){
        p[j+x]=1;
      }
      else{
        sum+=1;
      }
      //右下斜め判定
      if(q[j-x+(N-1)]==0){
        q[j-x+(N-1)]=1;
      }
      else{
        sum+=1;
      }
    } 

    //解を発見したらstring型でvectorに保存
    if(sum==0){
      for(int i=0;i<N;i++){
        l<<A[i];
      }
      vector.push_back(l.str());

      //重複解を見つけて取り出す
      size = vector.size();
      for(int j2=0;j2<size-1;j2++){
        if(vector[j2]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }

      //右に90度回転した結果を入れる
      for(int j3=0;j3<N;j3++){
        for(int k=0;k<N;k++){
          if(A[k]==j3){
            newa[j3]=(N-1)-k;
            break;
          }

        }
      }
      for(int i=0; i<N; i++){
        m << newa[i];
      }
      vector.push_back(m.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j4=0;j4<size-1;j4++){
        if(vector[j4]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }
      //右に180度回転した結果を入れる
      for(int e=0;e<N;e++){
        for(int v=0;v<N;v++){
          if(newa[v]==e){
            newb[e]=(N-1)-v;
            break;
          }
        }
      } 

      for(int i=0; i<N; i++){
        z << newb[i];
      }
      vector.push_back(z.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j=0;j<size-1;j++){
        if(vector[j]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }
      //右に270度回転した結果を入れる
      for(int j3=0;j3<N;j3++){
        for(int k=0;k<N;k++){
          if(newb[k]==j3){
            newc[j3]=(N-1)-k;
            break;
          }

        }
      }
      for(int i=0; i<N; i++){
        m1 << newc[i];
      }
      vector.push_back(m1.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j4=0;j4<size-1;j4++){
        if(vector[j4]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }

      //左右反転した結果を入れる
      for(int x=0;x<N;x++){
        n << (N-1)-A[x];
      }
      vector.push_back(n.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j=0;j<size-1;j++){
        if(vector[j]==vector[size-1]){
          vector.pop_back();
          break;
        }
      } 

      //上下反転した結果を入れる
      for(int j=0;j<N;j++){
        h1 <<A[(N-1)-j];
      }
      vector.push_back(h1.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j=0;j<size-1;j++){
        if(vector[j]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }

      //90°回転して左右反転した結果を入れる
      for(int x=0;x<N;x++){
        o1 << (N-1)-newa[x];
      }
      vector.push_back(o1.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j5=0;j5<size-1;j5++){
        if(vector[j5]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }
      //180°回転して左右反転した結果を入れる
      for(int x=0;x<N;x++){
        o2 << (N-1)-newb[x];
      }
      vector.push_back(o2.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j5=0;j5<size-1;j5++){
        if(vector[j5]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }

      //90°回転させ上下反転した結果を入れる
      for(int j=0;j<N;j++){
        h2 <<newa[(N-1)-j];
      }
      vector.push_back(h2.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j=0;j<size-1;j++){
        if(vector[j]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }
    }

    sum=0;
  }
  no=1;
}

/**
*@param B[N] TypeBの駒の配置を保存している
*@param vector 解となった駒の配置情報を保存する
*/
void fcheckB(int B[N],std::vector<std::string>& vector){
  int x=0;
  int p[2*N-1];//右斜め上の配列
  int q[2*N-1];//右斜め下の配列
  int sum=0;
  int size=0;
  int no=0;
  int newb[N];
  std::ostringstream l;
  std::ostringstream m;
  std::ostringstream n;
  std::ostringstream o; 

  //配列の初期化
  for(int t=0;t<2*N-1;t++){
    p[t]=0;
    q[t]=0;
  }
  //配置なしの場合を確認
  for(int i=0;i<N;i++){
    if(B[i]==N+1){
      no=1;
      break;
    }
  }

  if(no == 0){
    //B[N]の解判定
    for(int j=0;j<N;j++){
      x=B[j];//コマの位置情報をxに代入
      //右上斜め判定
      if(p[j+x]==0){
        p[j+x]=1;
      }
      else{
        sum+=1;
      }
      //右下斜め判定
      if(q[j-x+(N-1)]==0){
        q[j-x+(N-1)]=1;
      }
      else{
        sum+=1;
      }
    }

    //解を発見したらstring型でvectorに保存
    if(sum==0){
      for(int i=0;i<N;i++){
        l<<B[i];
      }
      vector.push_back(l.str());

      //重複解を見つけて取り出す
      size = vector.size();
      for(int j2=0;j2<size-1;j2++){
        if(vector[j2]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }

      //右に90度回転した結果を入れる
      for(int j3=0;j3<N;j3++){
        for(int k=0;k<N;k++){
          if(B[k]==j3){
            newb[j3]=(N-1)-k;
            break;
          }

        }
      }
      for(int i=0; i<N; i++){
        m << newb[i];
      }
      vector.push_back(m.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j4=0;j4<size-1;j4++){
        if(vector[j4]==vector[size-1]){
          vector.pop_back();
          break;
        }
      } 

      //左右反転した結果を入れる
      for(int x=0;x<N;x++){
        n << (N-1)-B[x];
      }
      vector.push_back(n.str());
      //重複解を見つけて取り出す
      size = vector.size();
      for(int j=0;j<size-1;j++){
        if(vector[j]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }
      //90°回転して左右反転した結果を入れる
      for(int x=0;x<N;x++){
        o << (N-1)-newb[x];
      }
      vector.push_back(o.str());

      //重複解を見つけて取り出す
      size = vector.size();
      for(int j5=0;j5<size-1;j5++){
        if(vector[j5]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }   

    }
    sum=0;
  }
  no=1;
}
/**
*@param C[N] TypeCの駒の配置を保存している
*@param vector 解となった駒の配置情報を保存する
*/
void fcheckC(int C[N],std::vector<std::string>& vector){
  int x=0;
  int p[2*N-1];//右斜め上の配列
  int q[2*N-1];//右斜め下の配列
  int sum=0;
  int size=0;
  int no=0;
  std::ostringstream l;
  std::ostringstream n;

  //配列の初期化
  for(int t=0;t<2*N-1;t++){
    p[t]=0;
    q[t]=0;
  }
  //配置なしの場合を確認
  for(int i=0;i<N;i++){
    if(C[i]==N+1){
      no=1;
      break;
    }
  }
  if(no == 0){
    //B[N]の解判定
    for(int j=0;j<N;j++){ 

      x=C[j];//コマの位置情報をxに代入
      //右上斜め判定
      if(p[j+x]==0){
        p[j+x]=1;
      }
      else{
        sum+=1;
      }
      //右下斜め判定
      if(q[j-x+(N-1)]==0){
        q[j-x+(N-1)]=1;
      }
      else{
        sum+=1;
      }
    } 

    //解を発見したらstring型でvectorに保存
    if(sum==0){
      for(int i=0;i<N;i++){
        l<<C[i];
      }
      vector.push_back(l.str());

      //重複解を見つけて取り出す
      size = vector.size();
      for(int j2=0;j2<size-1;j2++){
        if(vector[j2]==vector[size-1]){
          vector.pop_back();
          break;
        }
      }
      //左右反転した結果を入れる
      for(int x=0;x<N;x++){
        n << (N-1)-C[x];
      }
      vector.push_back(n.str());

      //重複解を見つけて取り出す
      size = vector.size();
      for(int j=0;j<size-1;j++){
        if(vector[j]==vector[size-1]){
          vector.pop_back();
          break; 

        }
      }
    }
    sum=0;
  }
  no=1;
} 


int main(void){

  int Q[N/2];//1/4部分解作成用 
  int A[N];//TypeA判定用配列 
  int B[N];//TypeB判定用配列 
  int C[N];//TypeC判定用配列 
  int judgeA=0;//判定用変数 
  int judgeB=0;//判定用変数 
  int judgeC=0;//判定用変数
  int qcount=0;//初期に配置したクイーンの数 
  std::vector<std::string> vector;//解保存用配列 
  /*
  void init(int [N/2],int [N],int [N],int [N]);
  void partget(int [N/2],int [N],int [N],int [N],int &); 
  void partgetA(int [N],int &);
  void partgetB(int [N/2],int [N]); 
  void partgetC(int [N/2],int [N]);

  void checkA(int [N],int &);
  void fcheckA(int [N],std::vector<std::string>&); 
  void checkB(int [N],int &);
  void fcheckB(int [N],std::vector<std::string>&); 
  void checkC(int [N],int &);
  void fcheckC(int [N],std::vector<std::string>&); 
  void allputA(int [N]);
  void allputB(int [N]); 
  void allputC(int [N]);
  void print0(int [N/2],int [N]); 
  srand((unsigned)time(NULL));
  LARGE_INTEGER freq,time_start,time_end;//周波数、開始時間、終了時間

  QueryPerformanceFrequency(&freq); 
  QueryPerformanceCounter(&time_start);//時間計測開始
  */

  srand((unsigned)time(NULL));
  for(int i=0;i<MAX;i++){
    init(Q,A,B,C);//1/4部分解のおよび各Typeの初期化 
    partget(Q,A,B,C,qcount);//1/4部分解の作成 
    partgetA(A,qcount);//TypeA1/2まで作成 
    partgetB(Q,B);//TypeBを1/2まで作成 
    partgetC(Q,C);//TypeCを1/2まで作成 
    checkA(A,judgeA);
    checkB(B,judgeB); 
    checkC(C,judgeC); 
    if(judgeA!=0){
      allputA(A); 
      fcheckA(A,vector);
    }
    if(judgeB!=0){
      allputB(B); 
      fcheckB(B,vector);
    }
    if(judgeC!=0){
      allputC(C); 
      fcheckC(C,vector);
    }

  }
//QueryPerformanceCounter(&time_end);//計測時間停止 
  for(int i=0;i<vector.size();i++){
    std::cout << "解"<< i+1 << " : "<< vector[i] << std::endl;
  }
  printf("解の数:%d¥n",vector.size());
  //printf("処理時間:%d[ms]¥n",(time_end.QuadPart-time_start.QuadPart)*1000 / freq.QuadPart);
}
