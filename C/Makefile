# Makefile

OS=${shell uname}
ifeq ($(OS),Linux)
#CC=/usr/bin/clang
CC=gcc
CFLAGS=-W -Wall -O3 -std=c99 -lm -lpthread
FUNCTION=`pkg-config --libs --cflags OpenCL`
CPPFLAGS=-I/usr/include/CL
#LDFLAGS=-L/usr/lib64 -L/usr/lib64/libOpenCL.so -L/usr/lib
LDFLAGS=-L/usr/lib64 -L/usr/lib64/libOpenCL.so
endif

ifeq ($(OS),Darwin)
CFLAGS=-Wall -W -O3 -std=c99 -lm 
FUNCTION=-framework OpenCL
#clang
CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
LDFLAGS=-L/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/7.3.0/lib
CPPFLAGS=-I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/7.3.0/include

#CC=/usr/local/opt/llvm/bin/clang-5.0
#CPPFLAGS=-I/usr/local/opt/llvm\@5/lib
#LDFLAGS=-L/usr/local/opt/llvm\@5/include

#CC=/usr/local/Cellar/llvm/4.0.1/bin/clang
#LDFLAGS=-L/usr/local/Cellar/llvm/4.0.1/lib
#CPPFLAGS=-I/usr/local/Cellar/llvm/4.0.1/include
# 蠢倥ｌ縺壹↓
# export LLVM_CONFIG="/usr/local/Cellar/llvm/4.0.1/bin/llvm-config";
#gcc
#CC=/Library/Developer/CommandLineTools/usr/bin/gcc
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
#CC=/usr/local/Cellar/gcc\@5/5.5.0/bin/gcc-5
#LDFLAGS=-L/usr/local/Cellar/gcc\@5/5.5.0/lib
#CPPFLAGS=-I/usr/local/Cellar/gcc\@5/5.5_0/include
# gcc-4
#CC=/usr/local/Cellar/gcc\@4.9/4.9.4/bin/gcc-4.9
#LDFLAGS=-L/usr/local/Cellar/gcc\@4.9/4.9.4/lib
#CPPFLAGS=-I/usr/local/Cellar/gcc\@4.9/4.9.4/include
endif
RM=rm -fr



nq50=07_50NQueen
nq50: 07_50gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq50) 07_50gpu_queens.c 

nq49=07_49NQueen
nq49: 07_49gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq49) 07_49gpu_queens.c 

nq48=07_48NQueen
nq48: 07_48gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq48) 07_48gpu_queens.c 

nq47=07_47NQueen
nq47: 07_47gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq47) 07_47gpu_queens.c 

nq46=07_46NQueen
nq46: 07_46gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq46) 07_46gpu_queens.c 

nq45=07_45NQueen
nq45: 07_45gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq45) 07_45gpu_queens.c 

nq44=07_44NQueen
nq44: 07_44gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq44) 07_44gpu_queens.c 

nq43=07_43NQueen
nq43: 07_43gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq43) 07_43gpu_queens.c 

nq42=07_42NQueen
nq42: 07_42gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq42) 07_42gpu_queens.c 

nq41=07_41NQueen
nq41: 07_41gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq41) 07_41gpu_queens.c 

nq40=07_40NQueen
nq40: 07_40gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq40) 07_40gpu_queens.c 

nq38=07_38NQueen
nq38: 07_38gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq38) 07_38gpu_queens.c 

nq37=07_37NQueen
nq37: 07_37gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq37) 07_37gpu_queens.c 

nq36=07_36NQueen
nq36: 07_36gpu_queens.c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq36) 07_36gpu_queens.c 

nq35=07_35NQueen
nq35: $(nq35).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq35) $(nq35).c 

nq34=07_34NQueen
nq34: $(nq34).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) $(FUNCTION) -o $(nq34) $(nq34).c 

all:nq49 nq48 nq47 nq46 nq45 nq44 nq43 nq42 nq41 nq40 nq38 nq37 nq36 nq35 nq34 nq33 nq32 nq31 nq30 nq29 nq28 nq27 nq26 nq25 nq24 nq23 nq22 nq21 nq20 nq19 nq18 nq17 nq16 nq15t nq15 nq14 nq13 nq12 nq11 nq10 nq9 nq8 nq7 nq6 nq5 nq4 nq3 nq2 nq1

clean:
	$(RM) a.out $(nq1) $(nq2) $(nq3) $(nq4) $(nq5) $(nq6) $(nq7) $(nq8) $(nq9) $(nq10) $(nq11) $(nq12) $(nq13) $(nq14) $(nq15t) $(nq15) $(nq16) $(nq17) $(nq18) $(nq19) $(nq20) $(nq21) $(nq22) $(nq23) $(nq24) $(nq25) $(nq26) $(nq27) $(nq28) $(nq29) $(nq30) $(nq31) $(nq32) $(nq33) $(nq34) $(nq35) *Sol 07_36NQueen 07_37NQueen 07_38NQueen 07_40NQueen 07_41NQueen 07_42NQueen 07_43NQueen 07_44NQueen 07_45NQueen 07_46NQueen 07_47NQueen 07_48NQueen 07_49NQueen


nq33=07_33NQueen
nq33: $(nq33).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq33) $(nq33).c -lpthread

nq32=07_32NQueen
nq32: $(nq32).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq32) $(nq32).c -lpthread

nq31=07_31NQueen
nq31: $(nq31).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq31) $(nq31).c -lpthread

nq30=07_30NQueen
nq30: $(nq30).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq30) $(nq30).c -lpthread

nq29=07_29NQueen
nq29: $(nq29).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq29) $(nq29).c -lpthread

nq28=07_28NQueen
nq28: $(nq28).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq28) $(nq28).c -lpthread

nq27=07_27NQueen
nq27: $(nq27).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq27) $(nq27).c -lpthread

nq26=07_26NQueen
nq26: $(nq26).c 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CPPFLAGS) -o $(nq26) $(nq26).c -lpthread

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


