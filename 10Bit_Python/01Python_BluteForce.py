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


# 実行 
$ python <filename.py>

# 実行結果
1
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

2
 0 3 1 4 2
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

3
 1 3 0 2 4
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+

4
 1 4 2 0 3
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+

5
 2 0 3 1 4
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

6
 2 4 1 3 0
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+

7
 3 0 2 4 1
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

8
 3 1 4 2 0
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+

9
 4 1 3 0 2
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+

10
 4 2 0 3 1
+-+-+-+-+-+
| | |O| | |
+-+-+-+-+-+
| | | | |O|
+-+-+-+-+-+
| |O| | | |
+-+-+-+-+-+
| | | |O| |
+-+-+-+-+-+
|O| | | | |
+-+-+-+-+-+

size: 5 TOTAL: 10 UNIQUE: 2

"""

#
# グローバル変数
MAX=21  # ボードサイズ最大値
TOTAL=0 # 解
board=[0 for i in range(MAX)] # ボード配列格納用
# 
# ボードレイアウト出力
def printRecord(size):
  global TOTAL
  global board

  print(TOTAL)
  sEcho=""
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
# ブルートフォース版効き筋チェック
def check_bluteForce(size):
  global board
  for r in range(1,size,1):
    for i in range(r):
      if board[i]>=board[r]:
        val=board[i]-board[r]
      else:
        val=board[r]-board[i]
      if board[i]==board[r] or val==(r-i):
        return 0
  return 1
#
# ブルートフォース
def bluteForce(row,size):
  col=0
  global TOTAL
  global board
  if row==size:
    if check_bluteForce(size)==1:
      TOTAL=TOTAL+1
      printRecord(size)
  else:
    for col in range(size):
      board[row]=col
      bluteForce(row+1,size)

#
# 実行
size=5
bluteForce(0,size) # ブルートフォース
print("size:",size,"TOTAL:",TOTAL)
#
