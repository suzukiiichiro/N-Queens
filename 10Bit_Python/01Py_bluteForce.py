#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
ブルートフォース版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

"""
# pypyを使う場合はコメントを解除
# pypyで再帰が高速化できる
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens01:
  size:int
  aboard:list[int]
  count:int

  def __init__(self,size:int)->None:
    self.size=size;
    self.aboard=[0 for i in range(self.size)]
    self.count=0 

  def printout(self)->None:
    self.count+=1;
    print(self.count, end=": ");
    for i in range(self.size):
      print(self.aboard[i], end="");
    print("");

  def nqueens(self,row:int)->None:
    if row is self.size:
      self.printout();
    else:
      for i in range(self.size):
        self.aboard[row]=i;
        self.nqueens(row+1);

# 1.ブルートフォース　ちからまかせ探索
# $ python <filename>
# $ pypy <fileName>
# $ codon build -release <filename>
if __name__ == '__main__':
  NQueens01(5).nqueens(0)

