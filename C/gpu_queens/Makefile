# Makefile for nqueens2.c

# Linux ###
CC=/usr/bin/gcc


# mac #######################################################
# llvm
CC=/Library/Developer/CommandLineTools/usr/bin/gcc
#CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
#CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
LDFLAGS=-L/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/7.3.0/lib
CPPFLAGS=-I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/7.3.0/include

#CC=/usr/local/Cellar/llvm/4.0.1/bin/clang
#LDFLAGS=-L/usr/local/Cellar/llvm/4.0.1/lib
#CPPFLAGS=-I/usr/local/Cellar/llvm/4.0.1/include
# 忘れずに
# export LLVM_CONFIG="/usr/local/Cellar/llvm/4.0.1/bin/llvm-config";

# gcc-7
#CC=/usr/local/Cellar/gcc/7.2.0/bin/gcc-7
#LDFLAGS=-L/usr/local/Cellar/gcc/7.2.0/lib
#CPPFLAGS=-I/usr/local/Cellar/gcc/7.2.0/include

# gcc-6
#CC=/usr/local/Cellar/gcc\@6/6.4.0/bin/gcc-6
#LDFLAGS=-L/usr/local/Cellar/gcc\@6/6.4.0/lib
#CPPFLAGS=-I/usr/local/Cellar/gcc\@6/6.4.0/include
#
# gcc-5
#CC=/usr/local/Cellar/gcc\@5/5.4.0_1/bin/gcc-5
#LDFLAGS=-L/usr/local/Cellar/gcc\@5/5.4.0_1/lib
#CPPFLAGS=-I/usr/local/Cellar/gcc\@5/5.4.0_1/include

# gcc-4
#CC=/usr/local/Cellar/gcc\@4.9/4.9.4/bin/gcc-4.9
#LDFLAGS=-L/usr/local/Cellar/gcc\@4.9/4.9.4/lib
#CPPFLAGS=-I/usr/local/Cellar/gcc\@4.9/4.9.4/include


#デバッグ時
#CFLAGS=-Wall -W -O3 -std=c99 -pthread -lpthread -lm -framework OpenCL
CFLAGS=-Wall -W -O3 -pthread -lpthread -lm -framework OpenCL

#最終処理結果実行用
#CFLAGS=-O3
RM=rm -fr

# gcc-4
#LDFLAGS=-L
#CPPFLAGS=-I

# Linux #######################################################
#LDFLAGS=-L
#CPPFLAGS=-I

nq=gpu_queens
nq:$(nq).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq) *.c 

all:
	nq
clean:
	a.out	



