# Makefile for nqueens2.c

CC=gcc
CFLAGS=-Wall -O2
RM=rm -fr

#CPPFLAGS=-fopenmp -I/usr/local/opt/open-mpi/include
#LDFLAGS= -L/usr/local/opt/open-mpi/lib -lmpi

APP07_8=07_8NQueen
nq8: $(APP07_8).c 
	$(CC) $(CFLAGS) -o $(APP07_8) $(APP07_8).c 

all: nq8 nq7 nq6 nq5 nq4 nq3 nq2 nq1 nq02 nq01

clean:
	$(RM) nq01 nq02 $(APP07_1) $(APP07_2) $(APP07_3) $(APP07_4) $(APP07_5) $(APP07_6) $(APP07_7)$(APP07_8)

APP07_7=07_7NQueen
nq7: $(APP07_7).c 
	$(CC) $(CFLAGS) -o $(APP07_7) $(APP07_7).c 

APP07_6=07_6NQueen
nq6: $(APP07_6).c 
	$(CC) $(CFLAGS) -o $(APP07_6) $(APP07_6).c 

APP07_5=07_5NQueen
nq5: $(APP07_5).c 
	$(CC) $(CFLAGS) -o $(APP07_5) $(APP07_5).c 

APP07_4=07_4NQueen
nq4: $(APP07_4).c 
	$(CC) $(CFLAGS) -o $(APP07_4) $(APP07_4).c 

APP07_3=07_3NQueen
nq3: $(APP07_3).c 
	$(CC) $(CFLAGS) -o $(APP07_3) $(APP07_3).c 

APP07_2=07_2NQueen
nq2: $(APP07_2).c 
	$(CC) $(CFLAGS) -o $(APP07_2) $(APP07_2).c 

APP07_1=07_1NQueen
nq1: $(APP07_1).c 
	$(CC) $(CFLAGS) -o $(APP07_1) $(APP07_1).c 
	
nq02: nqueens2.c
	$(CC) $(CFLAGS) -o nq02  nqueens2.c

nq01: nqueens1.c
	$(CC) $(CFLAGS) -o nq01  nqueens1.c

