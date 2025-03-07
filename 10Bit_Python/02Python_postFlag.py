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
  def __init__(self):
    self.size=8;
    self.count=0;
    self.aboard=[0 for i in range(self.size)];
    self.fa=[0 for i in range(self.size)];
  def printout(self):
    self.count+=1;
    print(self.count,end=": ");
    for i in range(self.size):
      print(self.aboard[i],end="");
    print("");
  def nqueens(self,row:int):
    if row==self.size-1:
      self.printout();
    else:
      for i in range(self.size):
        self.aboard[row]=i;
        if self.fa[i]==0:
          self.fa[i]=1;
          self.nqueens(row+1);
          self.fa[i]=0;

# 2.配置フラグ
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
if __name__ == '__main__':
  NQueens02().nqueens(0);

