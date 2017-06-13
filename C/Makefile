# Makefile for nqueens2.c

CC=gcc
CFLAGS=-Wall -O2
RM=rm -fr

CPPFLAGS=-fopenmp -I/usr/local/opt/open-mpi/include
LDFLAGS= -L/usr/local/opt/open-mpi/lib -lmpi

nq9=07_09NQueen
nq9: $(nq9).c 
	$(CC) $(CFLAGS) -o $(nq9) $(nq9).c 

nq10=07_10NQueen
nq10: $(nq10).c 
	$(CC) $(CFLAGS) -o $(nq10) $(nq10).c 

all: nq10 nq9 nq8 nq7 nq6 nq5 nq4 nq3 nq2 nq1 nq02 nq01

clean:
	$(RM) a.out nq01 nq02 $(nq1) $(nq2) $(nq3) $(nq4) $(nq5) $(nq6) $(nq7) $(nq8) $(nq9) $(nq10)

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
	
nq02: nqueens2.c
	$(CC) $(CFLAGS) -o nq02  nqueens2.c

nq01: nqueens1.c
	$(CC) $(CFLAGS) -o nq01  nqueens1.c

