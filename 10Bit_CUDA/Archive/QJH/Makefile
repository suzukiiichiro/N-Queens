# 実行ファイル名
TARGET = 
#生成したい静的ライブラリ名
NAME = 
#コンパイル対象としたいプログラム
SRCS = 
# 生成したいオブジェクト名
OBJS = $(SRCS:.c=.o)
#コンパイラ
CC = gcc
# コンパイルオプション
CFLAGS = -O3 -Wall
# デバッグコンパイルオプション
CDEBUGFLAGS = -g
# インクルードファイルのパス指定
INC = -I../inc
# ライブラリファイルノパス指定
LIB =

all:
	$(CC) $(CFLAGS) nqueen_dg_QJH_odd.cpp -o odd
	$(CC) $(CFLAGS) nqueen_dg_QJH_even.cpp -o even

debug:
	$(CC) $(CDEBUGFLAGS) nqueen_dg_QJH_odd.cpp -o odd
	$(CC) $(CDEBUGFLAGS) nqueen_dg_QJH_even.cpp -o even

clean:
	rm -fr *.o


