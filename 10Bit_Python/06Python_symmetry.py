#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
対象解除版 Ｎクイーン

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
UNIQUE=0
COUNT2=0                        # ミラー
COUNT4=0
COUNT8=0
BOUND1=0
BOUND2=0
TOPBIT=0
ENDBIT=0
LASTMASK=0
SIDEMASK=0
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
# 対象解除法　対象解除ロジック
def symmetryOps(size):
  global BOUND1,BOUND2
  global TOPBIT,ENDBIT
  global COUNT2,COUNT4,COUNT8
  """
  ２．クイーンが右上角以外にある場合、
  (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルか
  ら180度回転)させても、さらに90度回転(オリジナルから270度回転)させてもオリ
  ジナルと同型になる。
  こちらに該当するユニーク解が属するグループの要素数は、左右反転させたパター
  ンを加えて２個しかありません。
  """
  if board[BOUND2]==1:
    ptn=2
    own=1
    while own<size:
      bit=1
      you=size-1
      while (board[you]!=ptn) and (board[own]>=bit):
        bit<<=1
        you-=1
      if board[own]>bit :
        return 
      if board[own]<bit :
        break
      own+=1
      ptn<<=1 
    #90度回転して同型なら180度回転も270度回転も同型である
    if own>size-1:
      COUNT2+=1
      if DISPLAY==1:
        printRecord_bitmap(size,1)
      return 
  """
  ２．クイーンが右上角以外にある場合、
    (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナル
    とは異なる。ただし、180度回転させた場合はオリジナルと同型になることも有り得
    る。こちらに該当するユニーク解が属するグループの要素数は、180度回転させて同
    型になる場合は４個(左右反転×縦横回転)
  """
  # 180度回転
  if board[size-1]==ENDBIT:
    you=size-1-1
    own=1
    while own<size:
      bit=1
      ptn=TOPBIT
      while (ptn!=board[you]) and (board[own]>=bit):
        bit<<=1
        ptn>>=1
      if board[own]>bit:
        return 
      if board[own]<bit:
        break
      own+=1
      you-=1
    #90度回転が同型でなくても180度回転が同型であることもある
    if own>size-1:
      COUNT4+=1
      if DISPLAY==1:
        printRecord_bitmap(size,1)
      return
  """
  ２．クイーンが右上角以外にある場合、
    (3)180度回転させてもオリジナルと異なる場合は、８個(左右反転×縦横回転×上下反転)
  """
  # 270度回転
  if board[BOUND1]==TOPBIT:
    ptn=TOPBIT>>1
    own=1
    while own<=size-1:
      bit=1
      you=0
      while (board[you]!=ptn) and (board[own]>=bit):
        bit<<=1
        you+=1
      if board[own]>bit:
        return 
      if board[own]<bit:
        break
      own+=1
      ptn>>=1
    COUNT8+=1
    if DISPLAY==1:
      printRecord_bitmap(size,1)
#
# 対象解除法　角にQがないときのバックトラック
def symmetry_backTrack(size,row,left,down,right):
  mask=(1<<size)-1
  bitmap=mask&~(left|down|right)
  if row==(size-1):
    if bitmap:
      if not  bitmap&LASTMASK:
        board[row]=bitmap   # Qを配置
        symmetryOps(size)   # 対象解除
  else:
    if row<BOUND1:          # 上部サイド枝刈り
      bitmap=bitmap|SIDEMASK
      bitmap=bitmap^SIDEMASK
    else:
      if row==BOUND2:       # 下部サイド枝刈り
        if not down&SIDEMASK:
          return
        if down&SIDEMASK!=SIDEMASK:
          bitmap=bitmap&SIDEMASK
    while bitmap:
      bit=-bitmap&bitmap
      bitmap=bitmap^bit
      board[row]=bit        # Qを配置
      symmetry_backTrack(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
#
# 対象解除法　角にQがあるときのバックトラック
def symmetry_backTrack_corner(size,row,left,down,right):
  global BOUND1
  global COUNT8
  mask=(1<<size)-1
  bitmap=mask&~(left|down|right)
  if row==(size-1):
    """
    １．クイーンが右上角にある場合、ユニーク解が属する
    グループの要素数は必ず８個(＝２×４)
    """
    if bitmap :
      board[row]=bitmap
      if DISPLAY :
        printRecord_bitmap(size,1)
      COUNT8+=1
  else:
    if row<BOUND1:    # 枝刈り
      bitmap=bitmap|2
      bitmap=bitmap^2
    while bitmap:
      bit=-bitmap&bitmap
      bitmap=bitmap^bit
      board[row]=bit
      symmetry_backTrack_corner(size,row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
#
# 対象解除法
def symmetry(size):
  global TOTAL,UNIQUE,COUNT2,COUNT4,COUNT8
  global BOUND1,BOUND2
  global TOPBIT,ENDBIT,LASTMASK,SIDEMASK
  TOTAL=UNIQUE=COUNT2=COUNT4=COUNT8=0
  mask=(1<<size)-1
  TOPBIT=1<<(size-1)
  ENDBIT=LASTMASK=SIDEMASK=0
  BOUND1=2
  BOUND2=0
  board[0]=1
  while BOUND1>1 and BOUND1<size-1 :
    if BOUND1<size-1:
      bit=1<<BOUND1
      board[1]=bit    # ２行目にQを配置
      # 角にQがあるときのバックトラック
      symmetry_backTrack_corner(size,2,(2|bit)<<1,1|bit,(2|bit)>>1)
    BOUND1+=1
  TOPBIT=1<<(size-1)
  ENDBIT=TOPBIT>>1
  SIDEMASK=TOPBIT|1
  LASTMASK=TOPBIT|1
  BOUND1=1
  BOUND2=size-2
  while BOUND1>0 and BOUND2<size-1 and BOUND1<BOUND2:
    if BOUND1<BOUND2:
      bit=1<<BOUND1
      board[0]=bit    # Qを配置
      # 角にQがないときのバックトラック
      symmetry_backTrack(size,1,bit<<1,bit,bit>>1)
    BOUND1+=1
    BOUND2-=1
    ENDBIT=ENDBIT>>1
    LASTMASK=LASTMASK<<1|LASTMASK|LASTMASK>>1
  UNIQUE=COUNT8+COUNT4+COUNT2
  TOTAL=COUNT8*8 + COUNT4*4 + COUNT2*2
#
# 実行
size=5
DISPLAY=1;              # 表示 1:出力する 0: 出力しない
symmetry(size)          # ６．対象解除法
print("size:",size,"TOTAL:",TOTAL,"COUNT2:",COUNT2,"COUNT4:",COUNT4,"COUNT8:",COUNT8)
#
