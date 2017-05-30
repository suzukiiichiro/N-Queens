# Makefile for nqueens2.c

CC=gcc
CFLAGS=-Wall -O2
RM=rm -fr

CPPFLAGS=-fopenmp -I/usr/local/opt/open-mpi/include
LDFLAGS= -L/usr/local/opt/open-mpi/lib -lmpi

APP07_6=07_6NQueen
nq07_6: $(APP07_6).c 
	$(CC) $(CFLAGS) -o $(APP07_6) $(APP07_6).c 

APP07_5=07_5NQueen
nq07_5: $(APP07_5).c 
	$(CC) $(CFLAGS) -o $(APP07_5) $(APP07_5).c 

APP07_4=07_4NQueen
nq07_4: $(APP07_4).c 
	$(CC) $(CFLAGS) -o $(APP07_4) $(APP07_4).c 

APP07_3=07_3NQueen
nq07_3: $(APP07_3).c 
	$(CC) $(CFLAGS) -o $(APP07_3) $(APP07_3).c 

APP07_2=07_2NQueen
nq07_2: $(APP07_2).c 
	$(CC) $(CFLAGS) -o $(APP07_2) $(APP07_2).c 

APP07_1=07_1NQueen
nq07_1: $(APP07_1).c 
	$(CC) $(CFLAGS) -o $(APP07_1) $(APP07_1).c 
	
nq2: nqueens2.c
	$(CC) $(CFLAGS) -o nq2  nqueens2.c

nq1: nqueens1.c
	$(CC) $(CFLAGS) -o nq1  nqueens1.c

clean:
	$(RM) nq1 nq2 $(APP07_1) $(APP07_2) $(APP07_3) $(APP07_4) $(APP07_5) $(APP07_6)
