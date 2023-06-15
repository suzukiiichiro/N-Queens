#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
ミラー版 Ｎクイーン

詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens

# 実行 
$ python <filename.py>

# 実行結果
1
 0 2 4 1 3
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+

2
 0 3 1 4 2
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+

3
 1 3 0 2 4
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+

4
 1 4 2 0 3
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+

5
 2 4 3 1 4
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+

size: 5 TOTAL: 10 COUNT2: 5

"""

#
# グローバル変数
MAX=21                          # ボードサイズ最大値
TOTAL=0                         # 解
COUNT2=0                        # ミラー
board=[0 for i in range(MAX)]   # ボード配列格納用
down=[0 for i in range(MAX)]    # 効き筋チェック
left=[0 for i in range(MAX)]    # 効き筋チェック
right=[0 for i in range(MAX)]   # 効き筋チェック
#
# ビットマップ版ボードレイアウト出力
def printRecord_bitmap(size,flag):
  global TOTAL
  global baord
  print(TOTAL)
  sEcho=""
  """
  ビットマップ版
     ビットマップ版からは、左から数えます
     上下反転左右対称なので、これまでの上から数える手法と
     rowを下にたどって左から数える方法と解の数に変わりはありません。
     0 2 4 1 3 
    +-+-+-+-+-+
    |O| | | | | 0
    +-+-+-+-+-+
    | | |O| | | 2
    +-+-+-+-+-+
    | | | | |O| 4
    +-+-+-+-+-+
    | |O| | | | 1
    +-+-+-+-+-+
    | | | |O| | 3
    +-+-+-+-+-+
  """
  if flag:
    for i in range(size):
      for j in range(size):
        if board[i]&1<<j:
          sEcho+=" " + str(j)
  else:
    """
    ビットマップ版以外
    (ブルートフォース、バックトラック、配置フラグ)
    上から数えます
       0 2 4 1 3 
      +-+-+-+-+-+
      |O| | | | |
      +-+-+-+-+-+
      | | | |O| |
      +-+-+-+-+-+
      | |O| | | |
      +-+-+-+-+-+
      | | | | |O|
      +-+-+-+-+-+
      | | |O| | |
      +-+-+-+-+-+
    """
    for i in range(size):
      sEcho+=" " + str(board[i])
  print(sEcho)

  print ("+",end="")
  for i in range(size):
    print("-",end="")
    if i<(size-1):
      print("+",end="")
  print("+")
  for i in range(size):
    print("|",end="")
    for j in range(size):
      if flag:
        if board[i]&1<<j:
          print("O",end="")
        else:
          print(" ",end="")
      else:
        if i==board[j]:
          print("O",end="")
        else:
          print(" ",end="")
      if j<(size-1):
        print("|",end="")
    print("|")
    if i in range(size-1):
      print("+",end="")
      for j in range(size):
        print("-",end="")
        if j<(size-1):
          print("+",end="")
      print("+")
  print("+",end="")
  for i in range(size):
    print("-",end="")
    if i<(size-1):
      print("+",end="")
  print("+")
  print("")
#
# ミラーソルバー
def mirror_solver(size,row,left,down,right):
  global COUNT2
  global TOTAL
  mask=(1<<size)-1
  bit=0
  bitmap=0
  if row==size:
    COUNT2+=1
    TOTAL=TOTAL+1 # ボードレイアウト出力用
    printRecord_bitmap(size,1)
  else:
    # Ｑが配置可能な位置を表す
    bitmap=mask&~(left|down|right)
    while bitmap:
      bit=-bitmap&bitmap  # 一番右のビットを取り出す
      bitmap=bitmap^bit   # 配置可能なパターンが一つずつ取り出される
      board[row]=bit      # Ｑを配置
      mirror_solver(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
#
# ミラー
def Mirror(size):
  global COUNT2
  global TOTAL
  mask=(1<<size)-1
  bit=0
  """
  if size%2 :
    limit=size//2-1
  else:
    limit=size//2
  """
  limit=size%2 if size/2-1 else size/2
  # pythonでは割り算の切り捨ては`//`です
  for i in range(size//2):  # 奇数でも偶数でも通過
    bit=1<<i
    board[0]=bit            # １行目にＱを配置
    mirror_solver(size,1,bit<<1,bit,bit>>1)
  if size%2:                # 奇数で通過
    # Pythonでの割り算の切り捨ては`//`です
    bit=1<<(size-1)//2
    board[0]=1<<(size-1)//2 # １行目の中央にＱを配置
    left=bit<<1
    down=bit
    right=bit>>1
    for i in range(limit):
      bit=1<<i
      mirror_solver(size,2,(left|bit)<<1,down|bit,(right|bit)>>1)
  TOTAL=COUNT2<<1          # 倍にする
#
# 実行
size=5
Mirror(size)         # ５．ミラー
print("size:",size,"TOTAL:",TOTAL,"COUNT2:",COUNT2)
#
