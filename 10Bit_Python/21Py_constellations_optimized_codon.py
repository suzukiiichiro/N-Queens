#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
コンステレーション版 最適化　Ｎクイーン

1. Zobrist Hash（Opt-09）の導入とその用途
ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
例えば「既に同じビットマスク状態を訪問したか」判定、もしくは部分盤面パターンのメモ化など。
🔵 結論 大失敗　解が合わない

「bitboard探索の高速実装」＋「visitedによる枝刈り」は両立しません。
bitboard＋全探索で十分な高速化になるので、visitedの枝刈りを外しましょう。

なぜ「Total解数」が減るのか？
bitboard探索状態は、「同じ部分状態」でも、その経路や分岐ごとに 独立した配置になるため、「ただのld,rd,col,rowだけで状態ハッシュを取ってvisitedに入れる」と…

どうすればいい？
「visitedによる状態再訪枝刈り」は**「必ず探索木が本当に冗長な場合のみ」「十分に分岐条件がユニークな場合のみ」**に限定する　→ N-Queensでは、基本的には適用しない/もしくは部分最適化用途のみに留めるのが鉄則
元の「visitedなし」の設計に戻す（全探索） 　→ これで「全解」が正確に出ます
もし状態メモ化や枝刈りをやる場合は、　「盤面全体」「経路全体の履歴」まで情報量を増やす必要あり（でもメモリ爆発しやすい）
本来たどるべき別の正解経路も、「既に訪問済み」と見なして枝刈りしてしまう
→ 「重複解の排除」ではなく「正しい解の一部も捨てている」状態

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

fedora$ codon build -release 20Py_constellations_optimized_codon.py
fedora$ ./20Py_constellations_optimized_codon
 N:        Total       Unique        hh:mm:ss.ms
 5:           18            0         0:00:00.000
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          344            0         0:00:00.000
10:          700            0         0:00:00.001
11:         2440            0         0:00:00.003
12:        12088            0         0:00:00.009
13:        59726            0         0:00:00.059
14:       284772            0         0:00:00.354
15:      1664642            0         0:00:03.535
16:      9906686            0         0:01:02.645


19Py_constellations_optimized_codon.py は、非常に高度に最適化されており、コンステレーション法 + bit演算 + 対称性判定 + 回転ミラー除去（jasmin）+ Codon対応 という強力な設計です。

実装済（確認済み）の最適化手法

✅ bit演算による cols/hills/dales 衝突除去
✅ 左右対称・中央列特別処理（gen_constellations）
✅ jasmin() による「ミラー＋90度回転」済み（完成盤の正規化）
✅ symmetry() によるCOUNT2/4/8分類

タグ    対応課題    コード中のおおよその位置
[Opt-01]    ビット演算枝刈り（cols/hills/dales）    backtrack() の `free = mask & ~(cols
[Opt-02]    左右対称性除去（1 行目左半分）    solve_nqueens() の first_cols = range(n // 2)
[Opt-03]    中央列特別処理（奇数 N）    center_col = n // 2 if (n % 2 == 1)
[Opt-04]    180°対称除去    classify() / symmetries()（最終分類時 or 途中の簡易チェック）
[Opt-05]    角位置（col==0）分岐 & 対称分類（COUNT2/4/8）    solve_nqueens() で is_corner=True を渡す / classify()
[Opt-06]    並列処理（初手ごと）    Pool.imap_unordered(_worker, args)
[Opt-07]❎  行目以外でも部分対称除去    is_partial_canonical()（stub）を backtrack() 冒頭で呼ぶ
[Opt-08]❎  軽量 is_canonical の実装 & キャッシュ    is_partial_canonical() の中身を最適化 / @lru_cache / zobrist
[Opt-09]❎  Zobrist Hash    init_zobrist() と is_partial_canonical() 内のメモ化
[Opt-10]❎  マクロチェス（局所パターン）    violate_macro_patterns() を backtrack() で呼ぶ
[Opt-11]    構築時「ミラー＋90°回転」重複排除    （現状未実装・推奨しない。入れるなら is_partial_canonical() で）


1. Zobrist Hash（Opt-09）の導入とその用途
ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
例えば「既に同じビットマスク状態を訪問したか」判定、もしくは部分盤面パターンのメモ化など。

🎯 効果
「全く同じビットボード状態」を再訪した場合に探索を省略し、重複計算を排除できます。
盤面状態（ld, rd, col, rowなど）の組をハッシュ化し、訪問済みをvisitedセットで管理します。

Zobrist Hashで“再訪問”探索枝を枝刈り
set_pre_queens() や探索関数の再帰時に、 現在の ld, rd, col, row などの状態から64bitハッシュ値を生成
辞書やsetで「既に訪問した状態ならreturn」するだけで“ループ冗長探索”を削減

# 1.state_hash関数（Codon/Python両対応）
def state_hash(ld: int, rd: int, col: int, row: int) -> int:
    # codon は 64bit int 算術も高速
    return (ld * 0x9e3779b9) ^ (rd * 0x7f4a7c13) ^ (col * 0x6a5d39e9) ^ row

# 2.solve などの関数でset()を使う
visited: set[int] = set()
self.set_pre_queens(ld, rd, col, k, l, 1, 3 if j==N-1 else 4, LD, RD, counter, constellations, N, preset_queens, visited)

# 3.visited セット（型注釈つき）を solveやmainで用意し渡す
visited: set[int] = set()
self.set_pre_queens(ld, rd, col, k, l, 1, 3 if j==N-1 else 4, LD, RD, counter, constellations, N, preset_queens, visited)

# 4.set_pre_queens の再帰先頭に挿入
def set_pre_queens(self, ld: int, rd: int, col: int, k: int, l: int, row: int, queens: int, LD: int, RD: int, counter: list, constellations: list, N: int, preset_queens: int, visited: set[int]) -> None:
    mask: int = (1 << N) - 1
    # 状態ハッシュによる探索枝の枝刈り
    h: int = state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    # ...（この後従来の処理を続ける）

"""


import random
from operator import or_
# from functools import reduce
from typing import List,Set,Dict
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
#
class NQueens17:
  def __init__(self)->None:
    pass
  # ---------------------------
  # 1. Zobrist Hash（Opt-09）の導入とその用途
  # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
  # 例えば「既に同じビットマスク状態を訪問したか」判定、もしくは部分盤面パターンのメモ化など。
  def state_hash(self,ld: int, rd: int, col: int, row: int) -> int:
      """単純な状態ハッシュ（高速かつ衝突率低めなら何でも可）"""
      if None in (ld, rd, col, row):
          return -1
      return (ld * 0x9e3779b9) ^ (rd * 0x7f4a7c13) ^ (col * 0x6a5d39e9) ^ row
  # ---------------------------
  def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    if row==endmark:
      tempcounter[0]+=1
      return
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=~(next_ld|next_rd|next_col) # オーバーフロー防止  # マスクを適用<<注意
      if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd0BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3)) #<<注意
      if next_free:
        self.SQd0B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd0BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd1BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N4:int=N-4
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1|(1<<N4))
      if next_free:
        self.SQd1B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd1BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd1B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    if row==endmark:
      tempcounter[0]+=1
      return
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)
      if next_free and (row+1>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd1BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost 1-bit
      free&=free-1  # Remove the processed bit
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if next_free:
        # Recursive call with updated values
        self.SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    # General case when row != mark1
    while free:
      bit:int=free&-free  # Extract the rightmost 1-bit
      # bit:int=-free&free  # Extract the rightmost 1-bit
      free&=free-1  # Remove the processed bit
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        # Recursive call with updated values
        self.SQd1BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd1BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    while row==mark2 and free:
      # Extract the rightmost available position
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=((ld|bit)<<2)|1,(rd|bit)>>2,col|bit
      next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)
      if next_free and (row+2>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    # General case when row != mark2
    while free:
      # Extract the rightmost available position
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      # Update diagonal and column occupancies
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      # Recursive call if there are available positions
      if next_free:
        self.SQd1BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd1BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3  # Precomputed value for performance
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|2|(1<<N3))
      if nextfree:
        self.SQd1B(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    # General case
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd1BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd1BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if nextfree:
        self.SQd1BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    # General case
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      # bit:int=-free&free  # Extract the rightmost available position
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd1BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd1BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark2 and free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      # Calculate the next free positions
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if nextfree:
        self.SQd1B((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      # Calculate the next free positions
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd1BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd2BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark1 and free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      # 次の free の計算
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2)
      # 再帰的に SQd2B を呼び出す
      if nextfree:
        self.SQd2B((ld|bit)<<3|2,(rd|bit)>>3|(1<<N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      # 次の free の計算
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      # 再帰的に SQd2BlkB を呼び出す
      if nextfree:
        self.SQd2BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd2BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N4:int=N-4
    while row==mark1 and free:
      bit:int=free&-free  # 最下位のビットを取得
      free&=free-1  # 使用済みのビットを削除
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1)
      if next_free:
        self.SQd2B(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free  # 最下位のビットを取得
      free&=free-1  # 使用済みのビットを削除
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd2BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark2 and free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if next_free:
        self.SQd2B(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    # 通常の処理
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd2BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    while row==mark1 and free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if next_free:
        self.SQd2BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd2BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    while row==mark2 and free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if next_free:
        self.SQd2B(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQd2BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  #
  def SQd2BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if nextfree:
        self.SQd2BlB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    # 通常の処理
    while free:
      bit:int=free&-free
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQd2BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQd2B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    # rowがendmarkの場合の処理
    if row==endmark:
      if (free&(~1))>0:
        tempcounter[0]+=1
      return
    # 通常の処理
    while free:
      bit:int=free&-free  # 最も下位の1ビットを取得
      free&=free-1  # 使用済みビットを削除
      # 次の左対角線、右対角線、列の状態を計算
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      # 次の自由な位置を計算
      nextfree=~((next_ld)|(next_rd)|(next_col))
      if nextfree and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
        self.SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    while row==mark2 and free:
      bit:int=free&-free
      free&=free-1
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if nextfree:
        self.SQBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark1 and free:
      bit:int=free&-free  # Isolate the rightmost 1 bit.
      free&=free-1  # Remove the isolated bit from free.
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if nextfree:
        self.SQBlBjrB((ld|bit)<<2,((rd|bit)>>2)|(1<<N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free  # Isolate the rightmost 1 bit.
      free&=free-1  # Remove the isolated bit from free.
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if nextfree:
        self.SQBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  #
  def SQBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    if row==jmark:
      free&=~1  # Clear the least significant bit (mark position 0 unavailable).
      ld|=1  # Mark left diagonal as occupied for position 0.
      while free:
        bit:int=free&-free  # Get the lowest bit (first free position).
        free&=free-1  # Remove this position from the free positions.
        next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
        next_free:int=~((next_ld|next_rd|next_col))
        if next_free:
          self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
      return
    while free:
      bit:int=free&-free  # Get the lowest bit (first free position).
      free&=free-1  # Remove this position from the free positions.
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=~((next_ld|next_rd|next_col))
      if next_free:
        self.SQBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    if row==endmark:
      tempcounter[0]+=1
      return
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=~(next_ld|next_rd|next_col)&((1<<N)-1)
      if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
        self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      if next_free:
        self.SQBkBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  #
  def SQBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark2 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<N3))
      if next_free:
        self.SQBjrB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N4:int=N-4
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N4)|1)
      if next_free:
        self.SQBjrB(((ld|bit)<<3)|1,((rd|bit)>>3)|(1<<N4),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N3:int=N-3
    while row==mark1 and free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|(1<<N3)|2)
      if next_free:
        self.SQBjrB(((ld|bit)<<3)|2,((rd|bit)>>3)|(1<<N3),col|bit,row+3,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBjlBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBkBlBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBjlBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBjlBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBlBkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBjlBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBjlBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBklBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
          self.SQBjlBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def SQBjlBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    N1:int=N-1
    if row==N1-jmark:
      rd|=1<<N1
      free&=~(1<<N1)
      # if next_free:
      self.SQBlkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
      return
    while free:
      bit:int=free&-free
      # bit:int=-free&free
      free&=free-1
      next_free:int=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      if next_free:
        self.SQBjlBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,tempcounter,N,visited)
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i
    rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k)
    rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)
    return any(rot in ijkl_list for rot in (rot90, rot180, rot270))
  def symmetry(self,ijkl:int,N:int)->int:
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8
  def symmetry90(self,ijkl:int,N:int)->bool:
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    return (i<<15)+(j<<10)+(k<<5)+l
  def rot90(self,ijkl:int,N:int)->int:
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)
  def mirvert(self,ijkl:int,N:int)->int:
    return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))
  def ffmin(self,a:int,b:int)->int:
    return min(a,b)
  def geti(self,ijkl:int)->int:
    return (ijkl>>15)&0x1F
  def getj(self,ijkl:int)->int:
    return (ijkl>>10)&0x1F
  def getk(self,ijkl:int)->int:
    return (ijkl>>5)&0x1F
  def getl(self,ijkl:int)->int:
    return ijkl&0x1F
  def jasmin(self,ijkl:int,N:int)->int:
    # 最初の最小値と引数を設定
    arg=0
    min_val=self.ffmin(self.getj(ijkl),N-1-self.getj(ijkl))
    # i: 最初の行（上端） 90度回転2回
    if self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))<min_val:
      arg=2
      min_val=self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))
    # k: 最初の列（左端） 90度回転3回
    if self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))<min_val:
      arg=3
      min_val=self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))
    # l: 最後の列（右端） 90度回転1回
    if self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))<min_val:
      arg=1
      min_val=self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))
    # 90度回転を arg 回繰り返す
    for _ in range(arg):
      ijkl=self.rot90(ijkl,N)
    # 必要に応じて垂直方向のミラーリングを実行
    if self.getj(ijkl)<N-1-self.getj(ijkl):
      ijkl=self.mirvert(ijkl,N)
    return ijkl
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:set[int])->None:
    #------------------------------------
    h = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    #------------------------------------
    mask=(1<<N)-1  # setPreQueensで使用
    # k行とl行はスキップ
    if row==k or row==l:
      self.set_pre_queens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    if queens==preset_queens:
      constellation= {"ld": ld,"rd": rd,"col": col,"startijkl": row<<20,"solutions":0}
      # 新しいコンステレーションをリストに追加
      constellations.append(constellation)
      counter[0]+=1
      return
    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    while free:
      bit:int=free&-free  # 最も下位の1ビットを取得
      free&=free-1  # 使用済みビットを削除
      # クイーンを配置し、次の行に進む
      self.set_pre_queens((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)
  def exec_solutions(self,constellations:List[Dict[str,int]],N:int,visited:set[int])->None:
    jmark=j=k=l=ijkl=ld=rd=col=start_ijkl=start=free=LD=endmark=mark1=mark2=0
    small_mask=(1<<(N-2))-1
    temp_counter=[0]
    for constellation in constellations:
      # mark1,mark2=mark1,mark2
      start_ijkl=constellation["startijkl"]
      start=start_ijkl>>20
      ijkl=start_ijkl&((1<<20)-1)
      j,k,l=self.getj(ijkl),self.getk(ijkl),self.getl(ijkl)
      # 左右対角線と列の占有状況を設定
      ld,rd,col=constellation["ld"]>>1,constellation["rd"]>>1,(constellation["col"]>>1)|(~small_mask)
      LD=(1<<(N-1-j))|(1<<(N-1-l))
      ld|=LD>>(N-start)
      if start>k:
        rd|=(1<<(N-1-(start-k+1)))
      if j >= 2 * N-33-start:
        rd|=(1<<(N-1-j))<<(N-2-start)
      free=~(ld|rd|col)
      # 各ケースに応じた処理
      if j<(N-3):
        jmark,endmark=j+1,N-2
        if j>2 * N-34-start:
          if k<l:
            mark1,mark2=k-1,l-1
            if start<l:
              if start<k:
                if l!=k+1:
                  self.SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
                else: self.SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
              else: self.SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else: self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
          else:
            mark1,mark2=l-1,k-1
            if start<k:
              if start<l:
                if k!=l+1:
                  self.SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
                else: self.SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
              else: self.SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else: self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
        else:
          if k<l:
            mark1,mark2=k-1,l-1
            if l!=k+1:
              self.SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else: self.SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
          else:
            mark1,mark2=l-1,k-1
            if k != l+1:
              self.SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else: self.SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
      elif j==(N-3):
        endmark=N-2
        if k<l:
          mark1,mark2=k-1,l-1
          if start<l:
            if start<k:
              if l != k+1: self.SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
              else: self.SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else:
              mark2=l-1
              self.SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
          else: self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
        else:
          mark1,mark2=l-1,k-1
          endmark=N-2
          if start<k:
            if start<l:
              if k != l+1:
                self.SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
              else: self.SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else:
              mark2=k-1
              self.SQd2BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
          else: self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
      elif j==N-2: # クイーンjがコーナーからちょうど1列離れている場合
        if k<l:  # kが最初になることはない、lはクイーンの配置の関係で最後尾にはなれない
          endmark=N-2
          if start<l:  # 少なくともlがまだ来ていない場合
            if start<k:  # もしkもまだ来ていないなら
              mark1=k-1
              if l != k+1:  # kとlが隣り合っている場合
                mark2=l-1
                self.SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
              else: self.SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else:  # lがまだ来ていないなら
              mark2=l-1
              self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
          # すでにkとlが来ている場合
          else: self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
        else:  # l<k
          if start<k:  # 少なくともkがまだ来ていない場合
            if start<l:  # lがまだ来ていない場合
              if k<N-2:  # kが末尾にない場合
                mark1,endmark=l-1,N-2
                if k != l+1:  # lとkの間に空行がある場合
                  mark2=k-1
                  self.SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
                # lとkの間に空行がない場合
                else: self.SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
              else:  # kが末尾の場合
                if l != (N-3):  # lがkの直前でない場合
                  mark2,endmark=l-1,N-3
                  self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
                else:  # lがkの直前にある場合
                  endmark=N-4
                  self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
            else:  # もしkがまだ来ていないなら
              if k != N-2:  # kが末尾にない場合
                mark2,endmark=k-1,N-2
                self.SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
              else:  # kが末尾の場合
                endmark=N-3
                self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
          else: # kとlはスタートの前
            endmark=N-2
            self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
      else:  # クイーンjがコーナーに置かれている場合
        endmark=N-2
        if start>k:
          self.SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
        else: # クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用する方法
          mark1=k-1
          self.SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,temp_counter,N,visited)
      # 各コンステレーションのソリューション数を更新
      constellation["solutions"]=temp_counter[0] * self.symmetry(ijkl,N)
      temp_counter[0]=0
  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:set[int])->None:
    halfN=(N+1)//2  # Nの半分を切り上げ
    # コーナーにクイーンがいない場合の開始コンステレーションを計算する
    """
    for k in range(1,halfN):
      for l in range(k+1,N-1):
        for i in range(k+1,N-1):
          if i==(N-1)-l:
            continue
          for j in range(N-k-2,0,-1):
            if j==i or l==j:
              continue
            if not self.check_rotations(ijkl_list,i,j,k,l,N):
              ijkl_list.add(self.to_ijkl(i,j,k,l))
    """
    ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N-1) for i in range(k+1,N-1) if i != (N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N)
    )
    # コーナーにクイーンがある場合の開始コンステレーションを計算する
    ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N-1)})
    # Jasmin変換
    # ijkl_list_jasmin=set()
    # ijkl_list_jasmin.update(self.jasmin(start_constellation, N) for start_constellation in ijkl_list)
    ijkl_list_jasmin = {self.jasmin(c, N) for c in ijkl_list}

    ijkl_list=ijkl_list_jasmin
    L=1<<(N-1)  # Lは左端に1を立てる
    for sc in ijkl_list:
      i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      ld,rd,col=(L>>(i-1))|(1<<(N-k)),(L>>(i+1))|(1<<(l-1)),1|L|(L>>i)|(L>>j) 
      LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      counter=[0] # サブコンステレーションを生成
      #--------------------------
      self.set_pre_queens(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens,visited)
      #--------------------------
      current_size=len(constellations)
      # 生成されたサブコンステレーションにスタート情報を追加
      list(map(lambda target:target.__setitem__("startijkl",target["startijkl"]|self.to_ijkl(i,j,k,l)),(constellations[current_size-a-1] for a in range(counter[0]))))
class NQueens17_constellations():
  def main(self)->None:
    nmin:int=5
    nmax:int=18
    preset_queens:int=4  # 必要に応じて変更
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      NQ=NQueens17()
      #--------------------------
      visited:set[int]=set()
      #--------------------------
      NQ.gen_constellations(ijkl_list,constellations,size,preset_queens,visited)
      NQ.exec_solutions(constellations,size,visited)
      total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
if __name__=="__main__":
  NQueens17_constellations().main()
