#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" py01_nqueen.py """

#/**
# Pythonで学ぶアルゴリズムとデータ構造
# ステップバイステップでＮ−クイーン問題を最適化
# 一般社団法人    共同通信社    情報技術局    鈴木    維一郎(suzuki.iichiro@kyodone#ws.jp)
#
# 実行
# $ python py01_nqueen.py
#
#
# 1. ブルートフォース　力任せ探索
#
# 　全ての可能性のある解の候補を体系的に数え上げ、それぞれの解候補が問題の解とな
# るかをチェックする方法
#
#     (※)各行に１個の王妃を配置する組み合わせを再帰的に列挙組み合わせを生成するだ
#     けであって8王妃問題を解いているわけではありません
#
# 実行結果
# :
# :
# 16777209: 7 7 7 7 7 7 7 0
# 16777210: 7 7 7 7 7 7 7 1
# 16777211: 7 7 7 7 7 7 7 2
# 16777212: 7 7 7 7 7 7 7 3
# 16777213: 7 7 7 7 7 7 7 4
# 16777214: 7 7 7 7 7 7 7 5
# 16777215: 7 7 7 7 7 7 7 6
# 16777216: 7 7 7 7 7 7 7 7
#
# グローバル変数
MAX = 8
SIZE = 8                 #Nは8で固定
ABOARD = [0 for i in range(MAX)] #版の配列
COUNT = 0
#
def printout():
    """printout  """
    global COUNT              # pylint: disable=W0603
    COUNT += 1                # インクリメントはこのように書きます
    print(COUNT, end=": ")    # 改行したくないときは, を行末にいれます
    for i in range(SIZE):
        print(ABOARD[i], end="")
    print("")
#
def nqueen(row):
    """NQueen"""
    global ABOARD             # pylint: disable=W0603
    if row is SIZE:           # SIZEは8で固定
        printout()            # rowが8になったら出力
    else:
        for i in range(SIZE):
            ABOARD[row] = i
            nqueen(row+1)     #インクリメントしながら再帰

nqueen(0)     #ロジックメソッドを0を渡して呼び出し
#
