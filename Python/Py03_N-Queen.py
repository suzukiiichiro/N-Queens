#!/usr/bin/env python
# -*- coding: utf-8 -*-

# /**
#  Pythonで学ぶアルゴリズムとデータ構造
#  ステップバイステップでＮ−クイーン問題を最適化
#  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
# 
#  実行
#  $ python Py03_N-Queen.py
# 
# 
#  ３．バックトラック
# 
#  　各列、対角線上にクイーンがあるかどうかのフラグを用意し、途中で制約を満た
#  さない事が明らかな場合は、それ以降のパターン生成を行わない。
#  　各列、対角線上にクイーンがあるかどうかのフラグを用意することで高速化を図る。
#  　これまでは行方向と列方向に重複しない組み合わせを列挙するものですが、王妃
#  は斜め方向のコマをとることができるので、どの斜めライン上にも王妃をひとつだ
#  けしか配置できない制限を加える事により、深さ優先探索で全ての葉を訪問せず木
#  を降りても解がないと判明した時点で木を引き返すということができます。
# 
# 
#  実行結果
#   N:        Total       Unique        hh:mm:ss.ms
#   4:            2            0         0:00:00.000
#   5:           10            0         0:00:00.000
#   6:            4            0         0:00:00.000
#   7:           40            0         0:00:00.001
#   8:           92            0         0:00:00.004
#   9:          352            0         0:00:00.017
#  10:          724            0         0:00:00.077
#  11:         2680            0         0:00:00.380
#  12:        14200            0         0:00:02.090
#  13:        73712            0         0:00:12.024
#  14:       365596            0         0:01:13.316
#  15:      2279184            0         0:07:54.251
#  16:     14772512            0         0:54:46.040
#  17:     95815104            0         6:36:34.981
#  */
#
# グローバル変数
MAX=16; # N=15
TOTAL=0;
UNIQUE=0;
aBoard=[0 for i in range(MAX)];
fA=[0 for i in range(2*MAX-1)];	#縦列にクイーンを一つだけ配置
fB=[0 for i in range(2*MAX-1)];	#斜め列にクイーンを一つだけ配置
fC=[0 for i in range(2*MAX-1)];	#斜め列にクイーンを一つだけ配置
#
# ロジックメソッド
def NQueen(row,size):
  global fA;
  global fB;
  global fC;
  global TOTAL;
  if row==size:  #最後までこれたらカウント
    TOTAL+=1;
  else:
    for i in range(size):
      aBoard[row]=i;
      # 縦斜右斜左を判定
      if fA[i]==0 and fB[row-i+(size-1)]==0 and fC[row+i]==0 :
        fA[i]=fB[row-i+(size-1)]=fC[row+i]=1;
        NQueen(row+1,size);   #再帰
        fA[i]=fB[row-i+(size-1)]=fC[row+i]=0;
#
# メインメソッド
from datetime import datetime 
def main():
  global TOTAL;
  global UNIQUE;
  min=4;                          # Nの最小値（スタートの値）を格納
  print(" N:        Total       Unique        hh:mm:ss.ms");
  for i in range(min,MAX):
    TOTAL=0;
    UNIQUE=0;           # 初期化
    for j in range(i):
      aBoard[j]=j;      # 盤を初期化
    start_time = datetime.now() 
    NQueen(0,i)
    time_elapsed=datetime.now()-start_time 
    _text='{}'.format(time_elapsed)
    text=_text[:-3]
    print("%2d:%13d%13d%20s" % (i,TOTAL,UNIQUE,text)); # 出力
#
main();
#
