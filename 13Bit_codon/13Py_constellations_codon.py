#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
コンステレーション版Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ codon build -release 13Py_constellations_codon.py && ./13Py_constellations_codon
 N:        Total       Unique        hh:mm:ss.ms
 5:           18            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.001
12:        14200            0         0:00:00.002
13:        73712            0         0:00:00.012
14:       365596            0         0:00:00.056
15:      2279184            0         0:00:00.268
16:     14772512            0         0:00:01.665
17:     95815104            0         0:00:11.649
fedora$ 

13Py_constellations_codon.py（レビュー＆注釈つき）

要旨:
- "コンステレーション（星座）法" による N-Queens の分割探索。
- 既知の落とし穴：Python の無限精度整数での `~`（ビット反転）。**必ず N ビットでマスク**しないと、
  `free`/`next_free` が負数/巨大値になり、候補が常にあるように見えて桁あふれ→過少/過大計数の原因。

この修正版の主眼:
1) **すべての `~(...)` に N ビットマスク**を適用（`& ((1<<N)-1)`）。
2) `free` の初期化/更新も同様に**常にマスク**。
3) `(~(... ) > 0)` の判定は、`(mask&~( ... ))!=0` に置き換え、論理を明確化。
4) いくつかの変数名（`nextfree`/`next_free`）の表記ゆれを最小限で吸収（代入後に必ずマスク）。
5) 関数ごとに**目的コメント**を付与。

期待値: N=8 → 92,N=9 → 352,N=10 → 724,N=11 → 2680,N=12 → 14200,N=13 → 73712...
"""
from typing import List,Set,Dict
from datetime import datetime

# pypy を使う場合のみ有効化（Codon では無効）
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')


class NQueens13:
  def __init__(self)->None:
    pass

  # --- ユーティリティ ----------------------------------------------------
  @staticmethod
  def _maskN(x:int,N:int)->int:
    return x&((1<<N)-1)

  # i,j,k,l の packed/unpacked ユーティリティ
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    return (i<<15)+(j<<10)+(k<<5)+l

  def geti(self,ijkl:int)->int:
    return (ijkl>>15)&0x1F

  def getj(self,ijkl:int)->int:
    return (ijkl>>10)&0x1F

  def getk(self,ijkl:int)->int:
    return (ijkl>>5)&0x1F

  def getl(self,ijkl:int)->int:
    return ijkl&0x1F

  def rot90(self,ijkl:int,N:int)->int:
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)

  def mirvert(self,ijkl:int,N:int)->int:
    return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))

  @staticmethod
  def ffmin(a:int,b:int)->int:
    return a if a < b else b

  # --- 主要サブルーチン群（すべての next_free は N ビットでマスク） --------------
  def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    if row==endmark:
      tempcounter[0]+=1
      return
    free=self._maskN(free,N)
    while free:
      bit=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free=self._maskN(~(next_ld|next_rd|next_col),N)
      if next_free:
        lookahead=self._maskN(~(((next_ld<<1)|(next_rd>>1)|next_col)),N)
        if row >= endmark-1 or lookahead:
          self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd0BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N3=N-3
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)),N)
      if next_free:
        self.SQd0B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd0BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd1BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N4=N-4
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1|(1<<N4)),N)
      if next_free:
        self.SQd1B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd1BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd1B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    if row==endmark:
      tempcounter[0]+=1
      return
    free=self._maskN(free,N)
    while free:
      bit=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free=self._maskN(~(next_ld|next_rd|next_col),N)
      if next_free:
        lookahead=self._maskN(~(((next_ld<<1)|(next_rd>>1)|next_col)),N)
        if row+1 >= endmark or lookahead:
          self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd1BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N3=N-3
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)),N)
      if next_free:
        self.SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd1BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd1BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    free=self._maskN(free,N)
    while row==mark2 and free:
      bit=free&-free
      free&=free-1
      next_ld,next_rd,next_col=((ld|bit)<<2)|1,(rd|bit)>>2,col|bit
      next_free=self._maskN(~(next_ld|next_rd|next_col),N)
      if next_free:
        lookahead=self._maskN(~(((next_ld<<1)|(next_rd>>1)|next_col)),N)
        if row+2 >= endmark or lookahead:
          self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd1BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd1BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N3=N-3
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|2|(1<<N3)),N)
      if next_free:
        self.SQd1B(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd1BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd1BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1),N)
      if next_free:
        self.SQd1BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd1BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd1BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N3=N-3
    free=self._maskN(free,N)
    while row==mark2 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)),N)
      if next_free:
        self.SQd1B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd1BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd2BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N3=N-3
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2),N)
      if next_free:
        self.SQd2B((ld|bit)<<3|2,(rd|bit)>>3|(1<<N3),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd2BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd2BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N4=N-4
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1),N)
      if next_free:
        self.SQd2B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd2BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd2BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N3=N-3
    free=self._maskN(free,N)
    while row==mark2 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)),N)
      if next_free:
        self.SQd2B(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd2BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd2BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1),N)
      if next_free:
        self.SQd2BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd2BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd2BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    free=self._maskN(free,N)
    while row==mark2 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1),N)
      if next_free:
        self.SQd2B(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd2BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd2BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    N3=N-3
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)),N)
      if next_free:
        self.SQd2BlB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQd2BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQd2B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    mask=(1<<N)-1
    if row==endmark:
      if self._maskN(free&(~1),N):
        tempcounter[0]+=1
      return
    free=self._maskN(free,N)
    while free:
      bit=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free=self._maskN(~(next_ld|next_rd|next_col),N)
      if next_free:
        lookahead=self._maskN(~(((next_ld<<1)|(next_rd>>1)|(next_col))),N)
        if row >= endmark-1 or lookahead:
          self.SQd2B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  # 以降の SQB* 系も同様に `next_free` と `free` をマスクしつつそのまま踏襲
  def SQBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    free=self._maskN(free,N)
    while row==mark2 and free:
      bit=free&-free
      free&=free-1
      nextfree=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1),N)
      if nextfree:
        self.SQBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      nextfree=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if nextfree:
        self.SQBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N3=N-3
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      nextfree=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)),N)
      if nextfree:
        self.SQBlBjrB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      nextfree=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if nextfree:
        self.SQBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    free=self._maskN(free,N)
    if row==jmark:
      free&=~1
      ld |= 1
      while free:
        bit=free&-free
        free&=free-1
        next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
        next_free=self._maskN(~(next_ld|next_rd|next_col),N)
        if next_free:
          self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free=self._maskN(~(next_ld|next_rd|next_col),N)
      if next_free:
        self.SQBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    if row==endmark:
      tempcounter[0]+=1
      return
    free=self._maskN(free,N)
    while free:
      bit=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free=self._maskN(~(next_ld|next_rd|next_col),N)
      if next_free:
        lookahead=self._maskN(~(((next_ld<<1)|(next_rd>>1)|next_col)),N)
        if row >= endmark-1 or lookahead:
          self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1),N)
      if next_free:
        self.SQBkBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N3=N-3
    free=self._maskN(free,N)
    while row==mark2 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)),N)
      if next_free:
        self.SQBjrB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N4=N-4
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1),N)
      if next_free:
        self.SQBjrB(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N3=N-3
    free=self._maskN(free,N)
    while row==mark1 and free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2),N)
      if next_free:
        self.SQBjrB(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N)
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBjlBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N1=N-1
    free=self._maskN(free,N)
    if row==N1-jmark:
      rd |= 1<<N1
      free&=~(1<<N1)
      self.SQBkBlBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBjlBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBjlBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N1=N-1
    free=self._maskN(free,N)
    if row==N1-jmark:
      rd |= 1<<N1
      free&=~(1<<N1)
      self.SQBlBkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBjlBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBjlBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N1=N-1
    free=self._maskN(free,N)
    if row==N1-jmark:
      rd |= 1<<N1
      free&=~(1<<N1)
      self.SQBklBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBjlBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  def SQBjlBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:List[int],N:int)->None:
    N1=N-1
    free=self._maskN(free,N)
    if row==N1-jmark:
      rd |= 1<<N1
      free&=~(1<<N1)
      self.SQBlkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N)
      return
    while free:
      bit=free&-free
      free&=free-1
      next_free=self._maskN(~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)),N)
      if next_free:
        self.SQBjlBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N)

  # --- 対称性 -------------------------------------------------------------
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i
    rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k)
    rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)
    return any(rot in ijkl_list for rot in (rot90,rot180,rot270))

  def symmetry90(self,ijkl:int,N:int)->bool:
    return (
      (self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl)
    )==(
      ((N-1-self.getk(ijkl))<<15)
     +((N-1-self.getl(ijkl))<<10)
     +(self.getj(ijkl)<<5)
     +self.geti(ijkl)
    )

  def symmetry(self,ijkl:int,N:int)->int:
    if self.symmetry90(ijkl,N):
      return 2
    if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl):
      return 4
    return 8

  def jasmin(self,ijkl:int,N:int)->int:
    # 最小端に近い辺を基準に 90° 回転を規約化し、必要なら垂直ミラー
    arg=0
    min_val=self.ffmin(self.getj(ijkl),N-1-self.getj(ijkl))
    if self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl)) < min_val:
      arg=2
      min_val=self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))
    if self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl)) < min_val:
      arg=3
      min_val=self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))
    if self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl)) < min_val:
      arg=1
      min_val=self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))
    for _ in range(arg):
      ijkl=self.rot90(ijkl,N)
    if self.getj(ijkl) < N-1-self.getj(ijkl):
      ijkl=self.mirvert(ijkl,N)
    return ijkl

  # --- コンステレーションの前計算 -----------------------------------------
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:List[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    mask=(1<<N)-1
    if row==k or row==l:
      self.set_pre_queens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens)
      return
    if queens==preset_queens:
      constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":row<<20,"solutions":0})
      counter[0]+=1
      return
    free=self._maskN(~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row))),N)
    while free:
      bit=free&-free
      free&=free-1
      self.set_pre_queens((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens)

  def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
    small_mask=(1<<(N-2))-1
    temp_counter=[0]
    for constellation in constellations:
      start_ijkl=constellation["startijkl"]
      start=start_ijkl>>20
      ijkl=start_ijkl&((1<<20)-1)
      j,k,l=self.getj(ijkl),self.getk(ijkl),self.getl(ijkl)
      jmark=j+1      # 既定は j+1（多くの分岐でこの値を使う）
      mark1=0        # 未使用分岐でも未定義参照にならないよう初期化
      mark2=0
      ld=(constellation["ld"]>>1)
      rd=(constellation["rd"]>>1)
      col=(constellation["col"]>>1)|(~small_mask)
      LD=(1<<(N-1-j))|(1<<(N-1-l))
      ld |= LD>>(N-start)
      if start > k:
        rd |= (1<<(N-1-(start-k+1)))
      if j >= 2*N-33-start:
        rd |= (1<<(N-1-j))<<(N-2-start)
      # free の初期化は必ず N ビットでマスク
      free=self._maskN(~(ld|rd|col),N)
      if j < (N-3):
        jmark,endmark=j+1,N-2
        if j > 2*N-34-start:
          if k < l:
            mark1,mark2=k-1,l-1
            if start < l:
              if start < k:
                if l!=k+1:
                  self.SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
                else:
                  self.SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:
                self.SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            mark1,mark2=l-1,k-1
            if start < k:
              if start < l:
                if k!=l+1:
                  self.SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
                else:
                  self.SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:
                self.SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else:
          if k < l:
            mark1,mark2=k-1,l-1
            if l!=k+1:
              self.SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              self.SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            mark1,mark2=l-1,k-1
            if k!=l+1:
              self.SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              self.SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      elif j==(N-3):
        endmark=N-2
        if k < l:
          mark1,mark2=k-1,l-1
          if start < l:
            if start < k:
              if l!=k+1:
                self.SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:
                self.SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              mark2=l-1
              self.SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else:
          mark1,mark2=l-1,k-1
          endmark=N-2
          if start < k:
            if start < l:
              if k!=l+1:
                self.SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:
                self.SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              mark2=k-1
              self.SQd2BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      elif j==N-2:
        if k < l:
          endmark=N-2
          if start < l:
            if start < k:
              mark1=k-1
              if l!=k+1:
                mark2=l-1
                self.SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:
                self.SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              mark2=l-1
              self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else:
          if start < k:
            if start < l:
              if k < N-2:
                mark1,endmark=l-1,N-2
                if k!=l+1:
                  mark2=k-1
                  self.SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
                else:
                  self.SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:
                if l!=(N-3):
                  mark2,endmark=l-1,N-3
                  self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
                else:
                  endmark=N-4
                  self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
            else:
              if k!=N-2:
                mark2,endmark=k-1,N-2
                self.SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
              else:
                endmark=N-3
                self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
          else:
            endmark=N-2
            self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      else:
        endmark=N-2
        if start > k:
          self.SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
        else:
          mark1=k-1
          self.SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N)
      constellation["solutions"]=temp_counter[0]*self.symmetry(ijkl,N)
      temp_counter[0]=0

  # --- 初期星座（i,j,k,l）の列挙と正規化 -----------------------------------
  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    halfN=(N+1) // 2
    # 角に Q がない開始星座
    ijkl_list.update(
      self.to_ijkl(i,j,k,l)
      for k in range(1,halfN)
      for l in range(k+1,N-1)
      for i in range(k+1,N-1)
      if i!=(N-1)-l
      for j in range(N-k-2,0,-1)
      if j!=i and j!=l
      if not self.check_rotations(ijkl_list,i,j,k,l,N)
    )
    # 角に Q がある開始星座
    ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N-1)})
    # Jasmin 規約化
    ijkl_list={self.jasmin(c,N) for c in ijkl_list}

    L=1<<(N-1)
    for sc in ijkl_list:
      i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      ld=(L>>(i-1))|(1<<(N-k))
      rd=(L>>(i+1))|(1<<(l-1))
      col=1|L|(L>>i)|(L>>j)
      LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      counter=[0]
      self.set_pre_queens(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens)
      current_size=len(constellations)
      for a in range(counter[0]):
        constellations[current_size-a-1]["startijkl"] |= self.to_ijkl(i,j,k,l)


class NQueens13_constellations:

  def _bit_total(self,size:int)->int:
    # 小さなNは正攻法で数える（対称重みなし・全列挙）
    mask=(1<<size)-1
    total=0
    def bt(row:int,left:int,down:int,right:int):
      nonlocal total
      if row==size:
        total+=1
        return
      bitmap=mask&~(left|down|right)
      while bitmap:
        bit=-bitmap&bitmap
        bitmap^=bit
        bt(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
    bt(0,0,0,0)
    return total

  def main(self)->None:
    nmin:int=5
    nmax:int=18
    preset_queens:int=4
    print(" N:        Total       Unique         hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      if size <= 5:
        # ← フォールバック：N=5はここで正しい10を得る
        total=self._bit_total(size)
        dt=datetime.now()-start_time
        text=str(dt)[:-3]
        print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
        continue
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      nq=NQueens13()
      nq.gen_constellations(ijkl_list,constellations,size,preset_queens)
      nq.exec_solutions(constellations,size)
      total=sum(c["solutions"] for c in constellations if c["solutions"] > 0)
      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")

if __name__=="__main__":
  NQueens13_constellations().main()
