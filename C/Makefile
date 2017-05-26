# Makefile for nqueens2.c

CC=gcc
CFLAGS=-Wall -O2
RM=rm -fr

CPPFLAGS=-fopenmp -I/usr/local/opt/open-mpi/include
LDFLAGS= -L/usr/local/opt/open-mpi/lib -lmpi

nq2: nqueens2.c
	$(CC) $(CFLAGS) -o nq2  nqueens2.c

nq1: nqueens1.c
	$(CC) $(CFLAGS) -o nq1  nqueens1.c

clean:
	$(RM) nq1 nq2
