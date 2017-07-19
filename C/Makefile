# Makefile for nqueens2.c

CC=/usr/bin/gcc
#デバッグ時
CFLAGS=-Wall -W -O3 -std=c99 -pthread -lpthread -lm

#最終処理結果実行用
#CFLAGS=-O3
RM=rm -fr

#CPPFLAGS=-fopenmp -I/usr/local/opt/open-mpi/include
#CPPFLAGS=-fopenmp -I/usr/local/Cellar/open-mpi/2.1.1/include
#LDFLAGS= -L/usr/local/opt/open-mpi/lib -lmpi
#LDFLAGS= -L/usr/local/Cellar/open-mpi/2.1.1/lib -lmpi

#mac
LDFLAGS=-L/usr/local/opt/llvm/lib
CPPFLAGS=-I/usr/local/opt/llvm/include

nq25=07_25NQueen
nq25: $(nq25).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq25) $(nq25).c -lpthread

nq24=07_24NQueen
nq24: $(nq24).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq24) $(nq24).c -lpthread

nq23=07_23NQueen
nq23: $(nq23).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq23) $(nq23).c -lpthread

nq22=07_22NQueen
nq22: $(nq22).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq22) $(nq22).c -lpthread

nq21=07_21NQueen
nq21: $(nq21).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq21) $(nq21).c -lpthread

nq20=07_20NQueen
nq20: $(nq20).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq20) $(nq20).c -lpthread

nq19=07_19NQueen
nq19: $(nq19).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq19) $(nq19).c -lpthread

nq18=07_18NQueen
nq18: $(nq18).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq18) $(nq18).c -lpthread

nq17=07_17NQueen
nq17: $(nq17).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq17) $(nq17).c -lpthread

nq16=07_16NQueen
nq16: $(nq16).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq16) $(nq16).c -lpthread

nq15t=07_15NQueen_t
nq15t: $(nq15t).c 
	$(CC) $(CFLAGS) -o $(nq15t) $(nq15t).c 

nq15=07_15NQueen
nq15: $(nq15).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq15) $(nq15).c 

nq14=07_14NQueen
nq14: $(nq14).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq14) $(nq14).c 

nq13=07_13NQueen
nq13: $(nq13).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq13) $(nq13).c 

nq12=07_12NQueen
nq12: $(nq12).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq12) $(nq12).c 

nq11=07_11NQueen
nq11: $(nq11).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq11) $(nq11).c 

nq10=07_10NQueen
nq10: $(nq10).c 
	$(CC) $(CFLAGS) -o $(nq10) $(nq10).c 

nq9=07_09NQueen
nq9: $(nq9).c 
	$(CC) $(CFLAGS) -o $(nq9) $(nq9).c 

all:nq25 nq24 nq23 nq22 nq21 nq20 nq19 nq18 nq17 nq16 nq15t nq15 nq14 nq13 nq12 nq11 nq10 nq9 nq8 nq7 nq6 nq5 nq4 nq3 nq2 nq1

clean:
	$(RM) a.out $(nq1) $(nq2) $(nq3) $(nq4) $(nq5) $(nq6) $(nq7) $(nq8) $(nq9) $(nq10) $(nq11) $(nq12) $(nq13) $(nq14) $(nq15t) $(nq15) $(nq16) $(nq17) $(nq18) $(nq19) $(nq20) $(nq21) $(nq22) $(nq23) $(nq24) $(nq25)

nq8=07_08NQueen
nq8: $(nq8).c 
	$(CC) $(CFLAGS) -o $(nq8) $(nq8).c 

nq7=07_07NQueen
nq7: $(nq7).c 
	$(CC) $(CFLAGS) -o $(nq7) $(nq7).c 

nq6=07_06NQueen
nq6: $(nq6).c 
	$(CC) $(CFLAGS) -o $(nq6) $(nq6).c 

nq5=07_05NQueen
nq5: $(nq5).c 
	$(CC) $(CFLAGS) -o $(nq5) $(nq5).c 

nq4=07_04NQueen
nq4: $(nq4).c 
	$(CC) $(CFLAGS) -o $(nq4) $(nq4).c 

nq3=07_03NQueen
nq3: $(nq3).c 
	$(CC) $(CFLAGS) -o $(nq3) $(nq3).c 

nq2=07_02NQueen
nq2: $(nq2).c 
	$(CC) $(CFLAGS) -o $(nq2) $(nq2).c 

nq1=07_01NQueen
nq1: $(nq1).c 
	$(CC) $(CFLAGS) -o $(nq1) $(nq1).c 
	
#nq02: nqueens2.c
#	$(CC) $(CFLAGS) -o nq02  nqueens2.c
#
#nq01: nqueens1.c
#	$(CC) $(CFLAGS) -o nq01  nqueens1.c


