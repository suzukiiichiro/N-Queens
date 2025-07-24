#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ポストフラグ版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
Bash版ですが内容は同じです。

fedora$ python 02Py_postFlag.py
:
:
40312: 76542130
40313: 76542300
40314: 76542310
40315: 76543010
40316: 76543020
40317: 76543100
40318: 76543120
40319: 76543200
40320: 76543210

real	0m1.555s
user	0m0.745s
sys	0m0.475s
fedora$
"""
# pypyを使う場合はコメントを解除
# pypyで再帰が高速化できる
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens02:
  size:int
  count:int
  aboard:list[int]
  fa:list[int]
  def __init__(self,size:int)->None:
    self.size=size;
    self.count=0;
    self.aboard=[0 for i in range(self.size)];
    self.fa=[0 for i in range(self.size)];
  def printout(self)->None:
    self.count+=1;
    print(self.count,end=": ");
    for i in range(self.size):
      print(self.aboard[i],end="");
    print("");
  def nqueens(self,row:int)->None:
    if row==self.size-1:
      self.printout();
    else:
      for i in range(self.size):
        self.aboard[row]=i;
        if self.fa[i]==0:
          self.fa[i]=1;
          self.nqueens(row+1);
          self.fa[i]=0;
if __name__ == '__main__':
  NQueens02(8).nqueens(0);

